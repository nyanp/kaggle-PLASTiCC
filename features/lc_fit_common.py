from astropy.table import Table

def fitting(model, meta, data, object_id, zbounds='estimated', clip_bounds=False, t_bounds=False, snr=5):
    table = Table.from_pandas(data[data.object_id == object_id])

    if zbounds == 'estimated':
        z = meta.loc[object_id, 'hostgal_z_predicted']
        zerr = meta.loc[object_id, 'hostgal_photoz_err']
        photoz = meta.loc[object_id, 'hostgal_photoz']
        zerr = max(zerr, abs(z - photoz))
    elif zbounds == 'fixed':
        z = 0.7
        zerr = 0.7
    else:
        z = meta.loc[object_id, 'hostgal_photoz']
        zerr = meta.loc[object_id, 'hostgal_photoz_err']

    zmin = z - zerr
    zmax = z + zerr
    if clip_bounds:
        zmin = max(0.001, z - zerr)

    bounds = {
        'z': (zmin, zmax)
    }

    if t_bounds:
        tmin = data[data.object_id == object_id].mjd.min() - 50
        tmax = data[data.object_id == object_id].mjd.max()
        bounds['t0'] = (tmin, tmax)

    # run the fit
    result, fitted_model = sncosmo.fit_lc(
        table, model,
        model.param_names,  # parameters of model to vary
        bounds={'z': (zmin, zmax), 't0': (tmin, tmax)}, minsnr=snr)  # bounds on parameters (if any)

    return [result.chisq] + [result.ncall] + list(result.parameters) + list(result.errors.values())
