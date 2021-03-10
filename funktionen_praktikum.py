import pandas as pd
import intake
import seaborn as sns
import xarray as xr
import numpy as np
import dask
from dask.diagnostics import progress
from tqdm.autonotebook import tqdm
import fsspec
import cartopy.crs as ccrs
from matplotlib import pyplot as plt

def Was_gibt_es():
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    interessant=['Institute','Modelle','Variablen','Scenarien','Datensätze']
    interessant_cmip=['institution_id','source_id','variable_id','experiment_id','zstore']
    tmp=pd.DataFrame(data=None, columns=interessant, index=['Anzahl'])
    for ii,jj in enumerate(interessant):
        tmp[jj]=len(col.df[interessant_cmip[ii]].unique())
    return(tmp)

def Liste_aller_Variablen():
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    return(col.df['variable_id'].unique())

def Uebersicht_ausgewaehlter_Variablen():
    data=np.asarray([['Temperatur - 2m über dem Boden','Wolkenbedeckung','Aerosol Optische Dicke','Staub Aerosol Optische Dicke','Kohlenstoffdioxid','Methan','Windgeschwindigkeit - 10m über dem Boden', 'Blattflächenindex','Niederschlag','fester Niederschlag (Schnee)'],
         ['Amon','Amon','AERmon','AERmon','Amon','Amon','Amon','Lmon','Amon','Amon']]).T
    liste=pd.DataFrame(data=data, columns=['Bedeutung','table_id'], index=['tas','clt','od550aer','od550dust','co2','ch4','sfcWind','lai','pr','prsn'])
    return(liste)


def drop_all_bounds(ds):
    drop_vars = [vname for vname in ds.coords
                 if (('_bounds') in vname ) or ('_bnds') in vname]
    return ds.drop(drop_vars)

def open_dset(df):
    assert len(df) == 1
    ds = xr.open_zarr(fsspec.get_mapper(df.zstore.values[0]), consolidated=True)
    return drop_all_bounds(ds)

def open_delayed(df):
    return dask.delayed(open_dset)(df)

# calculate global means
def get_lat_name(ds):
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")

def global_mean(ds):
    lat = ds[get_lat_name(ds)]
    weight = np.cos(np.deg2rad(lat))
    weight /= weight.mean()
    other_dims = set(ds.dims) - {'time'}
    return (ds * weight).mean(other_dims)

def plot_verlauf_von(var,expts,start,end,table_id='Amon'):
    query = dict(
    experiment_id=expts,
    table_id=table_id,
    variable_id=[var],
    member_id = 'r1i1p1f1',
    )
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    col_subset = col.search(require_all_on=["source_id"], **query)
    
    from collections import defaultdict
    dsets = defaultdict(dict)
    
    for group, df in col_subset.df.groupby(by=['source_id', 'experiment_id']):
        dsets[group[0]][group[1]] = open_delayed(df)
    
    dsets_ = dask.compute(dict(dsets))[0]
    expt_da = xr.DataArray(expts, dims='experiment_id', name='experiment_id',
                       coords={'experiment_id': expts})

    dsets_aligned = {}
    
    print('Suchen und sortieren der angeforderten Daten.')
    
    for k, v in dsets_.items():
        expt_dsets = v.values()
        if any([d is None for d in expt_dsets]):
            print(f"Missing experiment for {k}")
            continue
    
        for ds in expt_dsets:
            ds.coords['year'] = ds.time.dt.year
    
        # workaround for
        # https://github.com/pydata/xarray/issues/2237#issuecomment-620961663
        dsets_ann_mean = [v[expt].pipe(global_mean)
                                 .swap_dims({'time': 'year'})
                                 .drop('time')
                                 .coarsen(year=12).mean()
                          for expt in expts]
    
        # align everything with the 4xCO2 experiment
        dsets_aligned[k] = xr.concat(dsets_ann_mean, join='outer',
                                     dim=expt_da)
        
    print('Berechnung des Globalenmittel von jedem einzelnen Modell.')
    print('Dies kann ein paar Minuten dauern.')
    with progress.ProgressBar():
        dsets_aligned_ = dask.compute(dsets_aligned)[0]
    
    source_ids = list(dsets_aligned_.keys())
    source_da = xr.DataArray(source_ids, dims='source_id', name='source_id',
                         coords={'source_id': source_ids})

    big_ds = xr.concat([ds.reset_coords(drop=True)
                    for ds in dsets_aligned_.values()],
                    dim=source_da)
    
    df_all = big_ds.sel(year=slice(start, end)).to_dataframe().reset_index()
    print('Erstellung des Plots.')
    sns.relplot(data=df_all,
            x="year", y=var, hue='experiment_id',
            kind="line", ci="sd", aspect=2);
    return()
    
    
def plot_30_Jahre_Klimatologie_von(var,scen,time,mod='GFDL-ESM4',table_id='Amon'):
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    query = dict(
        experiment_id=[scen],
        table_id=table_id,
        variable_id=[var],
        source_id=mod,#'MPI-ESM1-2-HR',
        member_id = 'r1i1p1f1',
    )
    map_data=col.search(require_all_on=["source_id"], **query)
    from collections import defaultdict
    dsets = defaultdict(dict)

    for group, df in map_data.df.groupby(by=['source_id', 'experiment_id']):
        dsets[group[0]][group[1]] = open_delayed(df)
    
    dsets_ = dask.compute(dict(dsets))[0]
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,4), subplot_kw={'projection': ccrs.Robinson()})
    (dsets_[mod][scen].sel(time=slice(str(time-15),str(time+15))).mean('time'))[var].plot(ax=axes ,transform=ccrs.PlateCarree(), cbar_kwargs=dict(shrink=0.5))
    axes.coastlines()

    return()

def plot_veraenderung_der_Klimatologie_von(var,scen1,scen2,time1,time2,mod='GFDL-ESM4',table_id='Amon',prozent=False):
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    query = dict(
        experiment_id=[scen1,scen2],
        table_id=table_id,
        variable_id=var,
        source_id=mod,#'MPI-ESM1-2-HR',
        member_id = 'r1i1p1f1',
    )
    map_data=col.search(require_all_on=["source_id"], **query)
    from collections import defaultdict
    dsets = defaultdict(dict)

    for group, df in map_data.df.groupby(by=['source_id', 'experiment_id']):
        dsets[group[0]][group[1]] = open_delayed(df)
    
    dsets_ = dask.compute(dict(dsets))[0]
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,4), subplot_kw={'projection': ccrs.Robinson()})
    if prozent:
        ((-dsets_[mod][scen1].sel(time=slice(str(time1-15),str(time1+15))).mean('time')+dsets_[mod][scen2].sel(time=slice(str(time2-15),str(time2+15))).mean('time'))/dsets_[mod][scen1].sel(time=slice(str(time1-15),str(time1+15))).mean('time')).where(dsets_[mod][scen1].sel(time=slice(str(time1-15),str(time1+15))).mean('time')>(0.0001*dsets_[mod][scen1].sel(time=slice(str(time1-15),str(time1+15))).mean('time')[var].max().values))[var].plot(ax=axes
                                                                    ,transform=ccrs.PlateCarree(),
                                                                    cbar_kwargs=dict(shrink=0.5))
    else:
        ((-dsets_[mod][scen1].sel(time=slice(str(time1-15),str(time1+15))).mean('time')+dsets_[mod][scen2].sel(time=slice(str(time2-15),str(time2+15))).mean('time')))[var].plot(ax=axes
                                                                    ,transform=ccrs.PlateCarree(),
                                                                    cbar_kwargs=dict(shrink=0.5))
    axes.coastlines()
    plt.title('Change of the 30 year climatology from '+str(time1)+' to '+str(time2)+ ' of \n'+var+' from the model '+mod)
    return()