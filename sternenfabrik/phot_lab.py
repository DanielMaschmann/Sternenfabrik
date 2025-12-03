"""
Tool to visualize PHANGS imaging data
"""
# technical functions
import os.path
import numpy as np
from scipy.interpolate import interp1d
# astropy functions
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import QTable, Table, Column
# plotting functions
import matplotlib.pyplot as plt
# own packages
from werkzeugkiste import helper_func, phys_params, phot_tools
from obszugang import ObsAccess, ObsTools, ClusterCatAccess
from malkasten import plotting_tools
from sternenstaub import DustTools


class PhotLab(ObsAccess):
    """
    Class to gather all functions to characterize photometric observations
    """

    def __init__(self, target_name=None, phot_hst_target_name=None, phot_hst_ha_cont_sub_target_name=None,
                 phot_nircam_target_name=None, phot_miri_target_name=None, phot_astrosat_target_name=None,
                 x_target_name=None,
                 nircam_data_ver='v1p1p1', miri_data_ver='v1p1p1', astrosat_data_ver='v1p0'):
        ObsAccess.__init__(
            self, target_name=target_name, phot_hst_target_name=phot_hst_target_name,
            phot_hst_ha_cont_sub_target_name=phot_hst_ha_cont_sub_target_name,
            phot_nircam_target_name=phot_nircam_target_name, phot_miri_target_name=phot_miri_target_name,
            phot_astrosat_target_name=phot_astrosat_target_name, x_target_name=x_target_name,
            nircam_data_ver=nircam_data_ver, miri_data_ver=miri_data_ver, astrosat_data_ver=astrosat_data_ver)

    def get_aperture_radii_and_scales(self, band_list, instrument_list, obs_list, roi_arcsec, bkg_roi_rad_in_arcsec,
                                      bkg_roi_rad_out_arcsec):

        standard_phot_aperture_arcsec_list = []
        phot_aperture_pix_list = []
        phot_aperture_arcsec_list = []
        bkg_rad_in_arcsec_list = []
        bkg_rad_out_arcsec_list = []
        fwhm_pix_list = []
        std_pix_list = []

        for band_idx, band in enumerate(band_list):

            # get psf scales for band
            # psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument=instrument_list[band_idx])
            # fwhm_arcsec = psf_dict['gaussian_fwhm']
            fwhm_arcsec = phot_tools.PSFTools.get_obs_psf_fwhm(band=band, instrument=instrument_list[band_idx])
            std_arcsec = phot_tools.PSFTools.get_obs_psf_std(band=band, instrument=instrument_list[band_idx])

            # get the standard aperture radii of the current band
            wcs = getattr(self, '%s_bands_data' % obs_list[band_idx])['%s_wcs_img' % band]
            fwhm_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=fwhm_arcsec, wcs=wcs).value
            std_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=std_arcsec, wcs=wcs).value
            # estimate the standard_aperture_radius
            standard_apert_rad_arcsec = phot_tools.ApertTools.get_standard_ap_rad_arcsec(obs=obs_list[band_idx],
                                                                                         band=band, wcs=wcs)
            print(standard_apert_rad_arcsec)
            # now check if the standard aperture is smaller than the radius of interest
            if standard_apert_rad_arcsec < roi_arcsec:
                # in this case we will simply compute the photometry inside this larger aperture
                phot_aperture_arcsec = roi_arcsec
                rad_in_arcsec, rad_out_arcsec = bkg_roi_rad_in_arcsec, bkg_roi_rad_out_arcsec
            else:
                phot_aperture_arcsec = standard_apert_rad_arcsec
                rad_in_arcsec, rad_out_arcsec = phot_tools.ApertTools.get_standard_bkg_annulus_rad_arcsec(
                    obs=obs_list[band_idx], band=band, wcs=wcs)

            phot_aperture_pix = helper_func.CoordTools.transform_world2pix_scale(
                length_in_arcsec=phot_aperture_arcsec, wcs=wcs).value

            standard_phot_aperture_arcsec_list.append(standard_apert_rad_arcsec)
            phot_aperture_pix_list.append(phot_aperture_pix)
            phot_aperture_arcsec_list.append(phot_aperture_arcsec)
            bkg_rad_in_arcsec_list.append(rad_in_arcsec)
            bkg_rad_out_arcsec_list.append(rad_out_arcsec)
            fwhm_pix_list.append(fwhm_pix)
            std_pix_list.append(std_pix)

        return (standard_phot_aperture_arcsec_list, phot_aperture_pix_list, phot_aperture_arcsec_list,
                bkg_rad_in_arcsec_list, bkg_rad_out_arcsec_list, fwhm_pix_list, std_pix_list)

    def get_src_cutout_and_recenter(self, ra, dec, band, instrument, roi_arcsec, max_rad_arcsec, cutout_size=None,
                                    re_centering=True):
        # get the cutout_size
        if cutout_size is None:
            cutout_size = (3 * max_rad_arcsec, 3 * max_rad_arcsec)
        else:
            if ((cutout_size[0] < 3 * max_rad_arcsec) |
                    (cutout_size[1] < 3 * max_rad_arcsec)):
                cutout_size = (3 * max_rad_arcsec, 3 * max_rad_arcsec)

        # get the cutout
        obs_cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size,
                                                    include_err=True, band_list=[band])

        # make sure there is data in the cutout!
        if obs_cutout_dict['%s_img_cutout' % band].data is None:
            return None

        # get a mask of bad values
        cutout_mask = np.isnan(obs_cutout_dict['%s_img_cutout' % band].data)

        # get the psf sizes as reference
        psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument=instrument)
        std_arcsec = psf_dict['gaussian_std']
        std_pix = helper_func.CoordTools.transform_world2pix_scale(
            length_in_arcsec=std_arcsec, wcs=obs_cutout_dict['%s_img_cutout' % band].wcs)

        ####################################
        #### re-centering of the source ####
        ####################################
        if re_centering:
            # first we perform a source detection
            # In order to do so we need simple statistics of the image.
            mean_cutout, median_cutout, std_cutout = sigma_clipped_stats(
                obs_cutout_dict['%s_img_cutout' % band].data, sigma=3.0)
            detected_peaks = phot_tools.SrcTools.detect_peaks(
                data=obs_cutout_dict['%s_img_cutout' % band].data,
                detection_threshold=median_cutout + 3 * std_cutout, box_size=3)

            # we re-center onto the brightest spot inside the ROI
            if detected_peaks is None:
                ra_re_center = ra
                dec_re_center = dec
            else:
                x_src = list(detected_peaks['x_peak'])
                y_src = list(detected_peaks['y_peak'])
                positions_world = obs_cutout_dict['%s_img_cutout' % band].wcs.pixel_to_world(
                    detected_peaks['x_peak'], detected_peaks['y_peak'])
                ra_src = list(positions_world.ra.deg)
                dec_src = list(positions_world.dec.deg)
                peak_values = detected_peaks['peak_value']

                src_dict = {'x_src': x_src, 'y_src': y_src, 'ra_src': ra_src, 'dec_src': dec_src,
                            'peak_value': peak_values}

                re_center_dict = phot_tools.SrcTools.re_center_src_world(
                    init_ra=ra, init_dec=dec, data=obs_cutout_dict['%s_img_cutout' % band].data,
                    wcs=obs_cutout_dict['%s_img_cutout' % band].wcs, mask=cutout_mask, src_dict=src_dict,
                    re_center_rad_arcsec=roi_arcsec,
                    centroid_rad_arcsec=std_arcsec * 3)
                ra_re_center = re_center_dict['ra_src_recenter']
                dec_re_center = re_center_dict['dec_src_recenter']
        else:
            ra_re_center = ra
            dec_re_center = dec

        return obs_cutout_dict, ra_re_center, dec_re_center

    @staticmethod
    def measure_morph_photometry(rad_profile_dict, psf_dict, img, bkg, img_err, wcs, ra, dec):

        # get average value in the PSF aperture:
        # print(psf_dict['gaussian_fwhm'])
        # print(psf_dict['gaussian_std'])

        radius_of_interes = psf_dict['gaussian_std'] * 3

        central_apert_stats_source = phot_tools.PhotTools.get_circ_apert_stats(data=img - bkg, data_err=img_err,
                                                                               wcs=wcs,
                                                                               ra=ra, dec=dec,
                                                                               aperture_rad=radius_of_interes)
        central_apert_stats_bkg = phot_tools.PhotTools.get_circ_apert_stats(data=bkg, data_err=img_err, wcs=wcs,
                                                                            ra=ra, dec=dec,
                                                                            aperture_rad=radius_of_interes)

        amp_list = []
        mu_list = []
        sig_list = []
        amp_err_list = []
        mu_err_list = []
        sig_err_list = []

        # fig, ax = plt.subplots(nrows=len(rad_profile_dict['slit_profile_dict']['list_angle_idx']) +1)

        for idx in rad_profile_dict['slit_profile_dict']['list_angle_idx']:
            mask_center = ((rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] > psf_dict[
                'gaussian_std'] * 3 * -1) &
                           (rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] < psf_dict[
                               'gaussian_std'] * 3))
            min_value_in_center = np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_center])
            max_value_in_center = np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_center])

            # ax[idx].plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            # ax[-1].plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            #
            # ax[idx].errorbar(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #          rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              yerr=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_err'],
            #          fmt='.')
            # plt.plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            # plt.show()

            mask_central_pixels = ((rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] >
                                    psf_dict['gaussian_std'] * -3) &
                                   (rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] <
                                    psf_dict['gaussian_std'] * 3))
            # ax[idx].scatter(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'][mask_central_pixels],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_central_pixels],
            #              color='red')

            # select the lower amplitude. There is a chance that the values are negative
            lower_amp = min_value_in_center
            upper_amp = max_value_in_center + np.abs(max_value_in_center * 2)

            # plt.plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')

            gaussian_fit_dict = helper_func.FitTools.fit_gauss(
                x_data=rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'][mask_central_pixels],
                y_data=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_central_pixels],
                y_data_err=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_err'][mask_central_pixels],
                amp_guess=max_value_in_center, mu_guess=0, sig_guess=psf_dict['gaussian_std'],
                lower_amp=lower_amp, upper_amp=upper_amp,
                lower_mu=psf_dict['gaussian_std'] * -5, upper_mu=psf_dict['gaussian_std'] * 5,
                lower_sigma=psf_dict['gaussian_std'], upper_sigma=psf_dict['gaussian_std'] * 5)

            # dummy_rad = np.linspace(np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']),
            #                         np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']), 500)
            # gauss = helper_func.FitTools.gaussian_func(
            #     amp=gaussian_fit_dict['amp'], mu=gaussian_fit_dict['mu'], sig=gaussian_fit_dict['sig'], x_data=dummy_rad)
            #
            # # ax[idx].plot(dummy_rad, gauss)
            # plt.plot(dummy_rad, gauss)
            # plt.show()

            amp_list.append(gaussian_fit_dict['amp'])
            mu_list.append(gaussian_fit_dict['mu'])
            sig_list.append(gaussian_fit_dict['sig'])

            amp_err_list.append(gaussian_fit_dict['amp_err'])
            mu_err_list.append(gaussian_fit_dict['mu_err'])
            sig_err_list.append(gaussian_fit_dict['sig_err'])

        # get the best matching gauss

        amp_list = np.array(amp_list)
        mu_list = np.array(mu_list)
        sig_list = np.array(sig_list)
        amp_err_list = np.array(amp_err_list)
        mu_err_list = np.array(mu_err_list)
        sig_err_list = np.array(sig_err_list)

        # get all the gaussian functions that are central
        mask_mu = np.abs(mu_list) < psf_dict['gaussian_std'] * 1
        mask_amp = (amp_list > 0)

        # if no function was detected in the center
        if sum(mask_mu * mask_amp) == 0:
            # non detection
            mean_amp = central_apert_stats_source.max
            mean_mu = 0
            mean_sig = psf_dict['gaussian_std']
            # get flux inside the 3 sigma aperture
            flux = central_apert_stats_source.sum
            flux_err = np.sqrt(central_apert_stats_source.sum_err ** 2 + central_apert_stats_bkg.sum_err ** 2)
            detect_flag = False
        else:
            # print(sum(mask_mu * mask_amp))
            # print(amp_list[mask_mu * mask_amp])
            # print(mu_list[mask_mu * mask_amp])
            # print(sig_list[mask_mu * mask_amp])

            mean_amp = np.mean(amp_list[mask_mu * mask_amp])
            mean_mu = np.mean(mu_list[mask_mu * mask_amp])
            mean_sig = np.mean(sig_list[mask_mu * mask_amp])

            mean_amp_err = np.mean(amp_err_list[mask_mu * mask_amp])
            mean_sig_err = np.mean(sig_err_list[mask_mu * mask_amp])

            # we need to do the sigma in pixel scale though
            mean_sig_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=mean_sig, wcs=wcs)
            # get gaussian integaral as flux
            flux = mean_amp * 2 * np.pi * (mean_sig_pix ** 2)
            flux_err = np.sqrt(mean_amp_err ** 2 * (2 * mean_sig_err) ** 2)
            flux_err = np.sqrt(flux_err ** 2 + central_apert_stats_bkg.sum_err ** 2)
            detect_flag = True

        dummy_rad = np.linspace(np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']),
                                np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']), 500)
        gauss = helper_func.FitTools.gaussian_func(
            amp=mean_amp, mu=mean_mu, sig=mean_sig, x_data=dummy_rad)

        photometry_dict = {
            'dummy_rad': dummy_rad,
            'gauss': gauss,
            'flux': flux,
            'flux_err': flux_err,
            'detect_flag': detect_flag
        }
        return photometry_dict

        # ax[-1].plot(dummy_rad, gauss)
        #
        #
        # plt.show()
        # # exit()

    def apert_phot_with_substructure(
            self, ra_list, dec_list, roi_arcsec, bkg_roi_rad_in_arcsec, bkg_roi_rad_out_arcsec, idx_list=None,
                                    band_list=None,
                                    include_hst=True,
                                    include_nircam=True,
                                    include_miri=True,
                                    include_astrosat=False,

                                    detect_sub_src=True,
                                    max_n_sub_src=10,
                                    psf_substructure_ratio_lim=2,
                                    substructure_roi_frac=0.9,
                                    flux_unit='mJy',
                                    verbose_flag=True,
                                    plot_flag=True,
                                    plot_output_path='plot_output/',
                                    ):

        # check if the coordinates are a list or just a float
        if isinstance(ra_list, float):
            ra_list = [ra_list]
            dec_list = [dec_list]

        if idx_list is not None:
            if isinstance(idx_list, float) | isinstance(idx_list, int):
                idx_list = [idx_list]
        else:
            idx_list = np.arange(len(ra_list))

        # get band list
        band_list, hst_band_list, nircam_band_list, miri_band_list, astrosat_band_list = (
            ObsTools.split_obs_band_list(
                band_list=band_list, hst_target_name=self.phot_hst_target_name,
                nircam_target_name=self.phot_nircam_target_name, miri_target_name=self.phot_miri_target_name,
                astrosat_target_name=self.phot_astrosat_target_name,
                            nircam_data_ver=self.nircam_data_ver, miri_data_ver=self.miri_data_ver,
                astrosat_data_ver=self.astrosat_data_ver,
                            include_hst=include_hst, include_nircam=include_nircam, include_miri=include_miri, include_astrosat=include_astrosat))

        # check if the bands are loaded!
        self.load_phangs_bands(band_list=band_list, flux_unit=flux_unit, load_err=True)

        # get observation abd instrument list
        obs_list, instrument_list, target_name_list = (
            ObsTools.get_obs_list(band_list=band_list, hst_band_list=hst_band_list, nircam_band_list=nircam_band_list,
                                  miri_band_list=miri_band_list, astrosat_band_list=astrosat_band_list,
                                  hst_target_name=self.phot_hst_target_name,
                                  nircam_target_name=self.phot_nircam_target_name,
                                  miri_target_name=self.phot_miri_target_name,
                                  astrosat_target_name=self.phot_astrosat_target_name,))

        (standard_phot_aperture_arcsec_list, phot_aperture_pix_list, phot_aperture_arcsec_list,
         bkg_rad_in_arcsec_list, bkg_rad_out_arcsec_list, fwhm_pix_list, std_pix_list) = (
            self.get_aperture_radii_and_scales(band_list=band_list, instrument_list=instrument_list, obs_list=obs_list,
                                               roi_arcsec=roi_arcsec, bkg_roi_rad_in_arcsec=bkg_roi_rad_in_arcsec,
                                               bkg_roi_rad_out_arcsec=bkg_roi_rad_out_arcsec))

        # check if substructure should be detected
        sub_structure_colname_list = []
        search_substructure_flag_list = []

        for band, fwhm_arcsec in zip(band_list, fwhm_pix_list):

            if roi_arcsec / (fwhm_arcsec / 2) > psf_substructure_ratio_lim:
                search_substructure_flag = True
                # add sub structure names
                for sub_structure_idx in range(max_n_sub_src):
                    sub_structure_colname_list.append('%s_sub_struct_ra_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_dec_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_flux_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_flux_err_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_peak_%i' % (band, sub_structure_idx))
                    sub_structure_colname_list.append('%s_sub_struct_median_bkg_%i' % (band, sub_structure_idx))
            else:
                search_substructure_flag = False
            search_substructure_flag_list.append(search_substructure_flag)

        flux_table_names = []
        for band in band_list:
            flux_table_names.append('%s_flux' % band)
            flux_table_names.append('%s_flux_err' % band)
            flux_table_names.append('%s_peak' % band)
            flux_table_names.append('%s_flux_flag' % band)
            flux_table_names.append('%s_median_bkg' % band)
            flux_table_names.append('%s_mean_bkg' % band)
            flux_table_names.append('%s_bkg_flag' % band)


        if verbose_flag: print('roi_arcsec ', roi_arcsec)


        #################################
        # now loop over all coordinates #
        #################################

        # construct a table

        r_rows = len(idx_list)
        n_cols = len(band_list) * 7

        flux_table = Table(np.zeros((r_rows, n_cols)), names=flux_table_names)

        sub_structure_data_table = Table(np.zeros((r_rows, len(sub_structure_colname_list))),
                                         names=sub_structure_colname_list)

        for running_idx, obj_idx, ra, dec in zip(range(len(idx_list)), idx_list, ra_list, dec_list):
            if verbose_flag: print(running_idx, obj_idx, ra, dec)

            if plot_flag:
                fontsize_small_label = 30
                fontsize_large_label = 40

                fig_size_individual = (5, 5)
                n_cols = 5
                n_rows = int(np.ceil((len(band_list)) / n_cols) + 2)

                fig = plt.figure(figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))
                gs = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.01, bottom=0.01, right=0.99, top=0.99,
                                      wspace=0.01, hspace=0.01)
                gs_sed = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.08, bottom=0.01, right=0.99, top=0.99,
                                          wspace=0.5, hspace=0.5)
                ax_sed = fig.add_subplot(gs_sed[:2, :])

            ###########################
            # now loop over all bands #
            ###########################
            idx_row = 2
            idx_col = 0
            for band_idx, band in enumerate(band_list):

                if not band in band_list:
                    continue

                if verbose_flag: print(band)
                # get the cutout_size
                img_cutout_size = (3 * bkg_rad_out_arcsec_list[band_idx], 3 * bkg_rad_out_arcsec_list[band_idx])
                # print(band, fwhm_arcsec, phot_aperture_arcsec, rad_in_arcsec, bkg_rad_out_arcsec_list[band_idx, img_cutout_size)

                obs_cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=img_cutout_size,
                                                            include_err=True, band_list=[band])
                if obs_cutout_dict['%s_img_cutout' % band].data is None:
                    continue

                forced_photomerty_dict = phot_tools.ApertTools.compute_apert_photometry(
                    apert_rad_arcsec=phot_aperture_arcsec_list[band_idx],
                    bkg_rad_annulus_in_arcsec=bkg_rad_in_arcsec_list[band_idx],
                    bkg_rad_annulus_out_arcsec=bkg_rad_out_arcsec_list[band_idx],
                    data=obs_cutout_dict['%s_img_cutout' % band].data,
                    data_err=obs_cutout_dict['%s_err_cutout' % band].data,
                    wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                    ra=ra, dec=dec, mask=None, sigma_clip_sig=3, sigma_clip_maxiters=10, sum_method='exact')

                flux_table['%s_flux' % band][running_idx] = forced_photomerty_dict['src_flux']
                flux_table['%s_flux_err' % band][running_idx] = forced_photomerty_dict['src_flux_err']
                flux_table['%s_peak' % band][running_idx] = forced_photomerty_dict['apert_max']
                flux_table['%s_flux_flag' % band][running_idx] = forced_photomerty_dict['flux_measure_ok_flag']
                flux_table['%s_median_bkg' % band][running_idx] = forced_photomerty_dict['bkg_median']
                flux_table['%s_mean_bkg' % band][running_idx] = forced_photomerty_dict['bkg_mean']
                flux_table['%s_bkg_flag' % band][running_idx] = forced_photomerty_dict['bkg_ok_flag']

                if plot_flag:

                    # plot img
                    ax_img = fig.add_subplot(gs[idx_row, idx_col],
                                             projection=obs_cutout_dict['%s_img_cutout' % band].wcs)
                    idx_col += 1
                    if idx_col >= n_cols:
                        idx_col = 0
                        idx_row += 1

                    mean, median, std = sigma_clipped_stats(obs_cutout_dict['%s_img_cutout' % band].data, sigma=3.0)
                    vmin = median - 1 * std
                    vmax = median + 30 * std

                    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
                    ax_img.imshow(obs_cutout_dict['%s_img_cutout' % band].data, norm=norm_img, cmap='Greys')
                    ax_img.axis('off')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_img, text=band.upper(),
                                                                   fontsize=fontsize_large_label,
                                                                   text_color='tab:orange', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top', path_eff=True,
                                                                   path_err_linewidth=3, path_eff_color='white',
                                                                   rotation=0)

                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img,
                                                                      pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                                      rad=phot_aperture_arcsec_list[band_idx],
                                                                      color='tab:red', line_style='-', line_width=3,
                                                                      alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img,
                                                                      pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                                      rad=bkg_rad_in_arcsec_list[band_idx],
                                                                      color='tab:blue', line_style='--', line_width=3,
                                                                      alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img,
                                                                      pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                                      rad=bkg_rad_out_arcsec_list[band_idx],
                                                                      color='tab:blue', line_style='--', line_width=3,
                                                                      alpha=1., fill=False)

                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_img, img_shape=obs_cutout_dict['%s_img_cutout' % band].data.shape,
                        wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                        bar_length=1, length_unit='arcsec', bar_color='red', text_color='red', line_width=4,
                        fontsize=fontsize_small_label,
                        va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01,
                        path_eff=True, path_eff_color='white')

                    # plot sed point
                    mean_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='mean_wave', unit='mu')
                    min_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='min_wave', unit='mu')
                    max_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='max_wave', unit='mu')

                    ax_sed.errorbar(mean_band_wavelength, forced_photomerty_dict['src_flux'],
                                    xerr=[[mean_band_wavelength - min_band_wavelength],
                                          [max_band_wavelength - mean_band_wavelength]],
                                    yerr=forced_photomerty_dict['src_flux_err'],
                                    fmt='.', color='k', ms=30)

                if search_substructure_flag_list[band_idx] & detect_sub_src:
                    # get cutout
                    # scale_map_cutout = helper_func.CoordTools.get_img_cutout(
                    #     img=scale_data, wcs=scale_wcs, coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg),
                    #     cutout_size=img_cutout_size)
                    # mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(obs_cutout_dict['%s_img_cutout' % band].data)
                    # source detection
                    sub_src_detect = phot_tools.SrcTools.detect_star_like_src(
                        data=obs_cutout_dict['%s_img_cutout' % band].data,
                        detection_threshold=forced_photomerty_dict['bkg_median'],
                        src_fwhm_pix=fwhm_pix_list[band_idx], min_separation=fwhm_pix_list[band_idx], roundhi=1,
                        roundlo=-1, sharphi=1.0, sharplo=0.2)
                    if sub_src_detect is None: continue

                    # central_coordx, central_coordy = obs_cutout_dict['%s_img_cutout' % band].data.shape[0] / 2, obs_cutout_dict['%s_img_cutout' % band].data.shape[1] / 2
                    central_coordx, central_coordy = obs_cutout_dict['%s_img_cutout' % band].wcs.world_to_pixel(
                        SkyCoord(ra=ra * u.deg, dec=dec * u.deg))
                    # sort sources by distance
                    dist2center = np.sqrt((sub_src_detect['xcentroid'] - central_coordx) ** 2 + (
                            sub_src_detect['ycentroid'] - central_coordy) ** 2)
                    sort = np.argsort(dist2center)
                    sub_src_detect = sub_src_detect[sort]
                    mask_src_in_apert = np.sqrt((sub_src_detect['xcentroid'] - central_coordx) ** 2 + (
                            sub_src_detect['ycentroid'] - central_coordy) ** 2) < (
                                                substructure_roi_frac * phot_aperture_pix_list[band_idx])

                    if sum(mask_src_in_apert) > 0:

                        coords_sub_src = obs_cutout_dict['%s_img_cutout' % band].wcs.pixel_to_world(
                            sub_src_detect['xcentroid'], sub_src_detect['ycentroid'])
                        ra_sub_src_list = coords_sub_src.ra.degree
                        dec_sub_src_list = coords_sub_src.dec.degree

                        native_rad_in_arcsec, native_rad_out_arcsec = phot_tools.ApertTools.get_standard_bkg_annulus_rad_arcsec(
                            obs=obs_list[band_idx], band=band, wcs=wcs)

                        for sub_structure_idx, ra_sub_src, dec_sub_src, src_x, src_y in zip(
                                range(len(ra_sub_src_list[mask_src_in_apert])), ra_sub_src_list[mask_src_in_apert],
                                dec_sub_src_list[mask_src_in_apert], sub_src_detect['xcentroid'][mask_src_in_apert],
                                sub_src_detect['ycentroid'][mask_src_in_apert]):

                            if sub_structure_idx >= (max_n_sub_src - 1):
                                continue

                            sub_phot_dict = phot_tools.ApertTools.compute_apert_photometry(
                                apert_rad_arcsec=standard_phot_aperture_arcsec_list[band_idx],
                                bkg_rad_annulus_in_arcsec=native_rad_in_arcsec,
                                bkg_rad_annulus_out_arcsec=native_rad_out_arcsec,
                                data=obs_cutout_dict['%s_img_cutout' % band].data,
                                data_err=obs_cutout_dict['%s_err_cutout' % band].data,
                                wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                                ra=ra_sub_src, dec=dec_sub_src, mask=None, sigma_clip_sig=3, sigma_clip_maxiters=5,
                                sum_method='exact')

                            sub_structure_data_table['%s_sub_struct_ra_%i' % (band, sub_structure_idx)][
                                running_idx] = ra_sub_src
                            sub_structure_data_table['%s_sub_struct_dec_%i' % (band, sub_structure_idx)][
                                running_idx] = dec_sub_src

                            sub_structure_data_table['%s_sub_struct_flux_%i' % (band, sub_structure_idx)][running_idx] = \
                                sub_phot_dict['src_flux']
                            sub_structure_data_table['%s_sub_struct_flux_err_%i' % (band, sub_structure_idx)][
                                running_idx] = sub_phot_dict['src_flux_err']

                            sub_structure_data_table['%s_sub_struct_peak_%i' % (band, sub_structure_idx)][running_idx] = \
                                sub_phot_dict['apert_max']
                            sub_structure_data_table['%s_sub_struct_median_bkg_%i' % (band, sub_structure_idx)][
                                running_idx] = sub_phot_dict['bkg_median']

                            if plot_flag:
                                ax_img.scatter(src_x, src_y, color='r', s=100)

                                ax_sed.errorbar(mean_band_wavelength, sub_phot_dict['src_flux'],
                                                xerr=[[mean_band_wavelength - min_band_wavelength],
                                                      [max_band_wavelength - mean_band_wavelength]],
                                                yerr=sub_phot_dict['src_flux_err'],
                                                fmt='.', color='tab:blue', ms=20)

            if plot_flag:
                ax_sed.set_xscale('log')
                ax_sed.set_yscale('log')
                ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fontsize_large_label, labelpad=-10)
                ax_sed.set_ylabel(r'flux [mJy]', fontsize=fontsize_large_label)
                ax_sed.tick_params(which='both', width=3, length=5, direction='in', color='k',
                                   labelsize=fontsize_large_label)

                # plt.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.05)
                fig.savefig(plot_output_path + 'complete_src_photmetry_%s_%i.png' % (self.phot_target_name, obj_idx))
                plt.close(fig)

        # remove empty columns
        for band_idx, band in enumerate(band_list):
            if not search_substructure_flag_list[band_idx]:
                continue
            for sub_structure_idx in range(max_n_sub_src):
                if np.all(sub_structure_data_table['%s_sub_struct_ra_%i' % (band, sub_structure_idx)] == 0):
                    sub_structure_data_table.remove_column('%s_sub_struct_ra_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_dec_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_flux_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_flux_err_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_peak_%i' % (band, sub_structure_idx))
                    sub_structure_data_table.remove_column('%s_sub_struct_median_bkg_%i' % (band, sub_structure_idx))

        return flux_table, sub_structure_data_table

    def compute_apert_corr_photometry(self, ra_list, dec_list,
                                      idx_list=None,
                                      band_list=None, flux_unit='mJy',
                                      verbose_flag=True,
                                      plot_flag=True,
                                      plot_output_path='plot_output/',
                                      ):

        # check if the coordinates are a list or just a float
        if isinstance(ra_list, float):
            ra_list = [ra_list]
            dec_list = [dec_list]

        if idx_list is not None:
            if isinstance(idx_list, float) | isinstance(idx_list, int):
                idx_list = [idx_list]
        else:
            idx_list = np.arange(len(ra_list))

        # get entire band list to fit
        complete_hst_band_list = ObsTools.get_hst_obs_band_list(target=self.phot_hst_target_name)
        complete_nircam_band_list = ObsTools.get_nircam_obs_band_list(target=self.phot_nircam_target_name,
                                                                                version=self.nircam_data_ver)
        complete_miri_band_list = ObsTools.get_miri_obs_band_list(target=self.phot_miri_target_name,
                                                                            version=self.miri_data_ver)

        # put together band list for the fitting
        if band_list is None:
            hst_band_list = complete_hst_band_list
            nircam_band_list = complete_nircam_band_list
            miri_band_list = complete_miri_band_list
            band_list = hst_band_list + nircam_band_list + miri_band_list
        else:
            hst_band_list = []
            nircam_band_list = []
            miri_band_list = []
            for band in band_list:
                if band in complete_hst_band_list:
                    hst_band_list.append(band)
                elif band in complete_nircam_band_list:
                    nircam_band_list.append(band)
                elif band in complete_miri_band_list:
                    miri_band_list.append(band)

        # check if the bands are loaded!
        self.load_phangs_bands(band_list=band_list, flux_unit=flux_unit, load_err=True)

        # get observation abd instrument list
        obs_list = []
        instrument_list = []
        target_name_list = []
        flux_table_names = []

        phot_aperture_pix_list = []
        phot_aperture_arcsec_list = []
        bkg_rad_in_arcsec_list = []
        bkg_rad_out_arcsec_list = []
        apert_corr_factor_list = []
        foreground_ext_mag_list = []

        for band in band_list:
            flux_table_names.append('%s_flux' % band)
            flux_table_names.append('%s_flux_err' % band)
            flux_table_names.append('%s_peak' % band)
            flux_table_names.append('%s_flux_flag' % band)
            flux_table_names.append('%s_median_bkg' % band)
            flux_table_names.append('%s_mean_bkg' % band)
            flux_table_names.append('%s_bkg_flag' % band)
            # get observations
            if band in hst_band_list:
                obs_list.append('hst')
                instrument_list.append(
                    ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band))
                target_name_list.append(self.phot_hst_target_name)
            elif band in nircam_band_list:
                obs_list.append('nircam')
                instrument_list.append('nircam')
                target_name_list.append(self.phot_nircam_target_name)
            elif band in miri_band_list:
                obs_list.append('miri')
                instrument_list.append('miri')
                target_name_list.append(self.phot_miri_target_name)
            else:
                raise RuntimeError(band, ' is in no observation present.')

            # get psf scales for band
            psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=band, instrument=instrument_list[-1])
            fwhm_arcsec = psf_dict['gaussian_fwhm']

            # get the standard aperture radii of the current band
            wcs = getattr(self, '%s_bands_data' % obs_list[-1])['%s_wcs_img' % band]

            # get standard photometry aperture
            phot_aperture_arcsec = phot_tools.ApertTools.get_standard_ap_rad_arcsec(obs=obs_list[-1],
                                                                                    band=band, wcs=wcs)
            # get aperture corr factor
            phot_apert_corr_fact = phot_tools.ApertTools.get_standard_ap_corr_fact(
                obs=obs_list[-1], band=band, target=target_name_list[-1])

            foreground_ext_mag = extinction_tools.ExtinctionTools.get_target_gal_ext_band(
                target=helper_func.FileTools.target_name_no_directions(target=target_name_list[-1]), obs=obs_list[-1],
                band=band)

            rad_in_arcsec, rad_out_arcsec = phot_tools.ApertTools.get_standard_bkg_annulus_rad_arcsec(
                obs=obs_list[-1], band=band, wcs=wcs)

            phot_aperture_pix = helper_func.CoordTools.transform_world2pix_scale(
                length_in_arcsec=phot_aperture_arcsec, wcs=wcs).value

            phot_aperture_pix_list.append(phot_aperture_pix)
            phot_aperture_arcsec_list.append(phot_aperture_arcsec)
            bkg_rad_in_arcsec_list.append(rad_in_arcsec)
            bkg_rad_out_arcsec_list.append(rad_out_arcsec)

            apert_corr_factor_list.append(phot_apert_corr_fact)
            foreground_ext_mag_list.append(foreground_ext_mag)

        #################################
        # now loop over all coordinates #
        #################################

        # construct a table

        r_rows = len(idx_list)
        n_cols = len(band_list) * 7

        flux_table = Table(np.zeros((r_rows, n_cols)), names=flux_table_names)

        for running_idx, obj_idx, ra, dec in zip(range(len(idx_list)), idx_list, ra_list, dec_list):
            if verbose_flag: print(running_idx, obj_idx, ra, dec)

            if plot_flag:
                fontsize_small_label = 30
                fontsize_large_label = 40

                fig_size_individual = (5, 5)
                n_cols = 5
                n_rows = int(np.ceil((len(band_list)) / n_cols) + 2)

                fig = plt.figure(figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))
                gs = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.01, bottom=0.01, right=0.99, top=0.99,
                                      wspace=0.01, hspace=0.01)
                gs_sed = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.08, bottom=0.01, right=0.99, top=0.99,
                                          wspace=0.5, hspace=0.5)
                ax_sed = fig.add_subplot(gs_sed[:2, :])

            ###########################
            # now loop over all bands #
            ###########################
            idx_row = 2
            idx_col = 0
            for band_idx, band in enumerate(band_list):

                if verbose_flag: print(band)
                # get the cutout_size
                img_cutout_size = (3 * bkg_rad_out_arcsec_list[band_idx], 3 * bkg_rad_out_arcsec_list[band_idx])
                # print(band, fwhm_arcsec, phot_aperture_arcsec, rad_in_arcsec, bkg_rad_out_arcsec_list[band_idx, img_cutout_size)

                obs_cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=img_cutout_size,
                                                            include_err=True, band_list=[band])
                if obs_cutout_dict['%s_img_cutout' % band].data is None:
                    continue

                apert_photomerty_dict = phot_tools.ApertTools.compute_apert_photometry(
                    apert_rad_arcsec=phot_aperture_arcsec_list[band_idx],
                    bkg_rad_annulus_in_arcsec=bkg_rad_in_arcsec_list[band_idx],
                    bkg_rad_annulus_out_arcsec=bkg_rad_out_arcsec_list[band_idx],
                    data=obs_cutout_dict['%s_img_cutout' % band].data,
                    data_err=obs_cutout_dict['%s_err_cutout' % band].data,
                    wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                    ra=ra, dec=dec, mask=None, sigma_clip_sig=3, sigma_clip_maxiters=10, sum_method='exact')

                # perform aperture and extinction correction

                print('src_flux ', apert_photomerty_dict['src_flux'])
                print('src_flux_err ', apert_photomerty_dict['src_flux_err'])
                print('apert_corr_factor_list[band_idx] ', foreground_ext_mag_list[band_idx])
                print('foreground_ext_mag_list[band_idx] ', foreground_ext_mag_list[band_idx],
                      10 ** (-foreground_ext_mag_list[band_idx] / -2.5))
                flux = apert_photomerty_dict['src_flux'] * apert_corr_factor_list[band_idx]
                flux_err = apert_photomerty_dict['src_flux_err'] * apert_corr_factor_list[band_idx]
                flux *= 10 ** (-foreground_ext_mag_list[band_idx] / -2.5)
                flux_err *= 10 ** (-foreground_ext_mag_list[band_idx] / -2.5)

                flux_table['%s_flux' % band][running_idx] = apert_photomerty_dict['src_flux']
                flux_table['%s_flux_err' % band][running_idx] = apert_photomerty_dict['src_flux_err']

                flux_table['%s_peak' % band][running_idx] = apert_photomerty_dict['apert_max']
                flux_table['%s_flux_flag' % band][running_idx] = apert_photomerty_dict['flux_measure_ok_flag']
                flux_table['%s_median_bkg' % band][running_idx] = apert_photomerty_dict['bkg_median']
                flux_table['%s_mean_bkg' % band][running_idx] = apert_photomerty_dict['bkg_mean']
                flux_table['%s_bkg_flag' % band][running_idx] = apert_photomerty_dict['bkg_ok_flag']

                if plot_flag:

                    # plot img
                    ax_img = fig.add_subplot(gs[idx_row, idx_col],
                                             projection=obs_cutout_dict['%s_img_cutout' % band].wcs)
                    idx_col += 1
                    if idx_col >= n_cols:
                        idx_col = 0
                        idx_row += 1

                    mean, median, std = sigma_clipped_stats(obs_cutout_dict['%s_img_cutout' % band].data, sigma=3.0)
                    vmin = median - 1 * std
                    vmax = median + 30 * std

                    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
                    ax_img.imshow(obs_cutout_dict['%s_img_cutout' % band].data, norm=norm_img, cmap='Greys')
                    ax_img.axis('off')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_img, text=band.upper(),
                                                                   fontsize=fontsize_large_label,
                                                                   text_color='tab:orange', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top', path_eff=True,
                                                                   path_err_linewidth=3, path_eff_color='white',
                                                                   rotation=0)

                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img,
                                                                      pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                                      rad=phot_aperture_arcsec_list[band_idx],
                                                                      color='tab:red', line_style='-', line_width=3,
                                                                      alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img,
                                                                      pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                                      rad=bkg_rad_in_arcsec_list[band_idx],
                                                                      color='tab:blue', line_style='--', line_width=3,
                                                                      alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_img,
                                                                      pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                                      rad=bkg_rad_out_arcsec_list[band_idx],
                                                                      color='tab:blue', line_style='--', line_width=3,
                                                                      alpha=1., fill=False)

                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_img, img_shape=obs_cutout_dict['%s_img_cutout' % band].data.shape,
                        wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                        bar_length=1, length_unit='arcsec', bar_color='red', text_color='red', line_width=4,
                        fontsize=fontsize_small_label,
                        va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01,
                        path_eff=True, path_eff_color='white')

                    # plot sed point
                    mean_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='mean_wave', unit='mu')
                    min_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='min_wave', unit='mu')
                    max_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='max_wave', unit='mu')

                    ax_sed.errorbar(mean_band_wavelength, apert_photomerty_dict['src_flux'],
                                    xerr=[[mean_band_wavelength - min_band_wavelength],
                                          [max_band_wavelength - mean_band_wavelength]],
                                    yerr=apert_photomerty_dict['src_flux_err'],
                                    fmt='.', color='k', ms=30)

            if plot_flag:
                ax_sed.set_xscale('log')
                ax_sed.set_yscale('log')
                ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fontsize_large_label, labelpad=-10)
                ax_sed.set_ylabel(r'flux [mJy]', fontsize=fontsize_large_label)
                ax_sed.tick_params(which='both', width=3, length=5, direction='in', color='k',
                                   labelsize=fontsize_large_label)

                # plt.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.05)
                fig.savefig(plot_output_path + 'complete_src_photmetry_%s_%i.png' % (self.phot_target_name, obj_idx))
                plt.close(fig)

        return flux_table

    def compute__apert_corr_photometry(
            self, ra_list, dec_list, roi_arcsec, bkg_roi_rad_in_arcsec, bkg_roi_rad_out_arcsec,
            idx_list=None,
            band_list=None,
            include_hst=True, include_nircam=True, include_miri=True, include_astrosat=False,

            profile_fit_rad_frac=1,
            cutout_size=None,
            obj_name=None,
            flux_unit='mJy',
            re_centering=True,
            verbose_flag=True,
            plot_flag=True,
            plot_output_path='plot_output/',
            save_as_pdf=False
    ):
        """
        This function is a first attempt in measuring an aperture corrected photometry of compact sources.

        """

        ########################
        #### organize input ####
        ########################
        # check if the coordinates are a list or just a float
        if isinstance(ra_list, float):
            ra_list = [ra_list]
            dec_list = [dec_list]
        # get an index for sources or make them up
        if idx_list is not None:
            if not isinstance(idx_list, list):
                idx_list = [idx_list]
        else:
            idx_list = np.arange(len(ra_list))

        # get band list
        band_list, hst_band_list, nircam_band_list, miri_band_list, astrosat_band_list = (
            ObsTools.split_obs_band_list(
                band_list=band_list, hst_target_name=self.phot_hst_target_name,
                nircam_target_name=self.phot_nircam_target_name, miri_target_name=self.phot_miri_target_name,
                astrosat_target_name=self.phot_astrosat_target_name,
                            nircam_data_ver=self.nircam_data_ver, miri_data_ver=self.miri_data_ver,
                astrosat_data_ver=self.astrosat_data_ver,
                            include_hst=include_hst, include_nircam=include_nircam, include_miri=include_miri, include_astrosat=include_astrosat))

        # check if the bands are loaded!
        self.load_phangs_bands(band_list=band_list, flux_unit=flux_unit, load_err=True)

        # get observation abd instrument list
        obs_list, instrument_list, target_name_list = (
            ObsTools.get_obs_list(band_list=band_list, hst_band_list=hst_band_list, nircam_band_list=nircam_band_list,
                                  miri_band_list=miri_band_list, astrosat_band_list=astrosat_band_list,
                                  hst_target_name=self.phot_hst_target_name,
                                  nircam_target_name=self.phot_nircam_target_name,
                                  miri_target_name=self.phot_miri_target_name,
                                  astrosat_target_name=self.phot_astrosat_target_name,))

        # get all the aperture and background sizes needed
        (standard_phot_aperture_arcsec_list, phot_aperture_pix_list, phot_aperture_arcsec_list,
         bkg_rad_in_arcsec_list, bkg_rad_out_arcsec_list, fwhm_pix_list, std_pix_list) = (
            self.get_aperture_radii_and_scales(band_list=band_list, instrument_list=instrument_list, obs_list=obs_list,
                                               roi_arcsec=roi_arcsec, bkg_roi_rad_in_arcsec=bkg_roi_rad_in_arcsec,
                                               bkg_roi_rad_out_arcsec=bkg_roi_rad_out_arcsec))

        # prepare flux table columns
        flux_table_names = []
        for band in band_list:
            flux_table_names.append('%s_flux' % band)
            flux_table_names.append('%s_flux_err' % band)
            flux_table_names.append('%s_apert_corr_fact' % band)
            flux_table_names.append('%s_apert_gal_ext_fact' % band)
            flux_table_names.append('%s_flux_bkg_sub_no_corr' % band)
            flux_table_names.append('%s_flux_apert' % band)
            flux_table_names.append('%s_peak' % band)
            flux_table_names.append('%s_flux_flag' % band)
            flux_table_names.append('%s_median_bkg' % band)
            flux_table_names.append('%s_mean_bkg' % band)
            flux_table_names.append('%s_bkg_flag' % band)
        # construct a table
        r_rows = len(idx_list)
        n_cols = len(band_list) * 11
        flux_table = Table(np.zeros((r_rows, n_cols)), names=flux_table_names)

        #################################
        # now loop over all coordinates #
        #################################
        for running_idx, obj_idx, ra, dec in zip(range(len(idx_list)), idx_list, ra_list, dec_list):
            if verbose_flag: print(running_idx, obj_idx, ra, dec)

            if plot_flag:
                fontsize_small_label = 40
                fontsize_large_label = 50

                fig_size_individual = (5, 5)
                n_cols = 7
                n_rows = int(np.ceil((len(band_list)) / n_cols) * 3 + 2)

                fig = plt.figure(figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))
                gs = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.01, bottom=0.01, right=0.99, top=0.99,
                                      wspace=0.01, hspace=0.01)
                gs_sed = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.08, bottom=0.01, right=0.99, top=0.99,
                                          wspace=0.5, hspace=0.5)
                ax_sed = fig.add_subplot(gs_sed[:2, :])

            ###########################
            # now loop over all bands #
            ###########################
            idx_row = 2
            idx_col = 0
            for band_idx, band in enumerate(band_list):

                if verbose_flag:
                    print(band)

                obs_cutout_dict, ra_re_center, dec_re_center = (
                    self.get_src_cutout_and_recenter(ra=ra, dec=dec, band=band,
                                                     instrument=instrument_list[band_idx], roi_arcsec=roi_arcsec,
                                                     max_rad_arcsec=bkg_rad_out_arcsec_list[band_idx], cutout_size=cutout_size,
                                    re_centering=re_centering))

                # perform photometry
                forced_photomerty_dict = phot_tools.ApertTools.compute_apert_photometry(
                    apert_rad_arcsec=phot_aperture_arcsec_list[band_idx],
                    bkg_rad_annulus_in_arcsec=bkg_rad_in_arcsec_list[band_idx],
                    bkg_rad_annulus_out_arcsec=bkg_rad_out_arcsec_list[band_idx],
                    data=obs_cutout_dict['%s_img_cutout' % band].data,
                    data_err=obs_cutout_dict['%s_err_cutout' % band].data,
                    wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                    ra=ra_re_center, dec=dec_re_center,
                    mask=None, sigma_clip_sig=3, sigma_clip_maxiters=10, sum_method='exact')

                # if the flux inside the aperture is negative there is no way we can do anything!.
                # This is an absolute non detection!
                if (forced_photomerty_dict['apert_flux'] <= 0) | (forced_photomerty_dict['src_flux'] <= 0):
                    neg_flx_in_apert = True
                else:
                    neg_flx_in_apert = False

                ########################################
                #### Perform an aperture correction ####
                ########################################
                # radial profile dict with also slit profiles
                rad_profile_dict = phot_tools.ProfileTools.get_rad_profile_dict(
                    img=obs_cutout_dict['%s_img_cutout' % band].data - forced_photomerty_dict['bkg_median'],
                    wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                    ra=ra_re_center, dec=dec_re_center,
                    n_slits=12, max_rad_arcsec=bkg_rad_out_arcsec_list[band_idx] * 1.1,
                    img_err=obs_cutout_dict['%s_err_cutout' % band].data,
                    img_mask=np.isnan(obs_cutout_dict['%s_img_cutout' % band].data -
                                      forced_photomerty_dict['bkg_median']))

                # only if there is a signal in the aperture
                if not neg_flx_in_apert:

                    rad_of_interest_pix = helper_func.CoordTools.transform_world2pix_scale(
                        length_in_arcsec=phot_aperture_arcsec_list[band_idx],
                        wcs=obs_cutout_dict['%s_img_cutout' % band].wcs)

                    amp_list, mu_list, sig_list, amp_err_list, mu_err_list, sig_err_list = (
                        phot_tools.ProfileTools.fit_gauss2rad_profiles(
                            rad_profile_dict=rad_profile_dict['slit_profile_dict'], std_pix=std_pix_list[band_idx],
                            upper_sig_fact=10, central_rad_fact=5,
                            radius_of_interest=rad_of_interest_pix * profile_fit_rad_frac))

                    # now select the best gaussian fit
                    best_gauss_dict = phot_tools.ProfileTools.get_best_gaussian_profile_params(
                        amp_list=amp_list, mu_list=mu_list, sig_list=sig_list,
                        peak_acceptance_rad_pix=std_pix_list[band_idx] * 1, n_good_fits_needed=4)

                    # find the best gaussian fit!
                    mean_mu_arcsec = helper_func.CoordTools.transform_pix2world_scale(
                        length_in_pix=best_gauss_dict['mean_mu'], wcs=obs_cutout_dict['%s_img_cutout' % band].wcs)
                    mean_sig_arcsec = helper_func.CoordTools.transform_pix2world_scale(
                        length_in_pix=best_gauss_dict['mean_sig'], wcs=obs_cutout_dict['%s_img_cutout' % band].wcs)

                    # get a dummy gaussian
                    n_pixels_in_bkg_rad = int(helper_func.CoordTools.transform_world2pix_scale(
                        length_in_arcsec=bkg_rad_out_arcsec_list[band_idx],
                        wcs=obs_cutout_dict['%s_img_cutout' % band].wcs) * 2)
                    dummy_rad_arcsec = np.linspace(-bkg_rad_out_arcsec_list[band_idx],
                                                   bkg_rad_out_arcsec_list[band_idx],
                                                   n_pixels_in_bkg_rad * 10)
                    mean_gauss = helper_func.FitTools.gaussian_func(
                        amp=best_gauss_dict['mean_amp'], mu=mean_mu_arcsec,
                        sig=mean_sig_arcsec, x_data=dummy_rad_arcsec)

                    print('mean_sig_arcsec ', mean_sig_arcsec)
                    # correction factor
                    corr_fact, too_extended_flag = phot_tools.PSFTools.get_apert_gauss_corr_fact(
                        band=band, instrument=instrument_list[band_idx], apert_rad=phot_aperture_arcsec_list[band_idx],
                        std=mean_sig_arcsec)

                else:
                    corr_fact = None

                # get galactic reddening correction
                fore_ground_ext = DustTools.get_target_gal_ext_band(
                    target=self.phot_target_name, obs=obs_list[band_idx], band=band, ext_law='G23')

                gal_red_corr_fact = 10 ** (fore_ground_ext / 2.5)

                if plot_flag:

                    ######################
                    ###### plot img ######
                    ######################
                    # add axis
                    ax_img = fig.add_subplot(gs[idx_row, idx_col],
                                             projection=obs_cutout_dict['%s_img_cutout' % band].wcs)
                    # get an image norm
                    mean, median, std = sigma_clipped_stats(obs_cutout_dict['%s_img_cutout' % band].data, sigma=3.0)
                    vmin = median - 1 * std
                    vmax = median + 30 * std
                    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
                    # display image
                    ax_img.imshow(obs_cutout_dict['%s_img_cutout' % band].data, norm=norm_img, cmap='Greys')
                    # cosmetics
                    ax_img.axis('off')
                    plotting_tools.StrTools.display_text_in_corner(
                        ax=ax_img, text=band.upper(), fontsize=fontsize_large_label, text_color='tab:red',
                        path_eff_color='k', )
                    coords_world = SkyCoord(ra=ra_re_center * u.deg, dec=dec_re_center * u.deg)
                    # add a scale bar
                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_img, img_shape=obs_cutout_dict['%s_img_cutout' % band].data.shape,
                        wcs=obs_cutout_dict['%s_img_cutout' % band].wcs, bar_length=1, length_unit='arcsec',
                        bar_color='red', text_color='red', line_width=4, fontsize=fontsize_small_label,
                        va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01,
                        path_eff=True, path_eff_color='k')
                    # plot aperture_radius and background annuli
                    plotting_tools.WCSPlottingTools.plot_coord_circle(
                        ax=ax_img, pos=coords_world, rad=phot_aperture_arcsec_list[band_idx],
                        color='tab:red', line_style='-', line_width=5, alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(
                        ax=ax_img, pos=coords_world, rad=bkg_rad_in_arcsec_list[band_idx],
                        color='tab:blue', line_style='--', line_width=5, alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(
                        ax=ax_img, pos=coords_world, rad=bkg_rad_out_arcsec_list[band_idx],
                        color='tab:blue', line_style='--', line_width=5, alpha=1., fill=False)

                    ##################################
                    ###### plot radial profiles ######
                    ##################################
                    # add axis
                    ax_profile = fig.add_subplot(gs[idx_row + 1, idx_col])

                    ax_profile.plot(rad_profile_dict['rad'], rad_profile_dict['profile'], linewidth=4, color='k')
                    ax_profile.fill_between(x=rad_profile_dict['rad'],
                                            y1=rad_profile_dict['profile'] + rad_profile_dict['profile_err'],
                                            y2=rad_profile_dict['profile'] - rad_profile_dict['profile_err'],
                                            color='lightgray')
                    max_rad_pofile = np.max(rad_profile_dict['profile'])

                    ax_profile.plot([phot_aperture_arcsec_list[band_idx], phot_aperture_arcsec_list[band_idx]],
                                    [0, max_rad_pofile], color='tab:red', linewidth=4)
                    ax_profile.plot([bkg_rad_in_arcsec_list[band_idx], bkg_rad_in_arcsec_list[band_idx]],
                                    [0, max_rad_pofile], color='tab:blue', linestyle='--',
                                    linewidth=4)
                    ax_profile.plot([bkg_rad_out_arcsec_list[band_idx], bkg_rad_out_arcsec_list[band_idx]],
                                    [0, max_rad_pofile], color='tab:blue', linestyle='--',
                                    linewidth=4)

                    ax_profile.set_xticklabels([])
                    ax_profile.set_yticklabels([])
                    plotting_tools.AxisTools.frame2axis(ax=ax_profile, color='k', line_width=3)
                    ax_profile.tick_params(which='both', width=3, length=5, direction='in', color='k',
                                           labelsize=fontsize_large_label)

                    ax_slit_profiles = fig.add_subplot(gs[idx_row + 2, idx_col])
                    # plot all the gaussians
                    for idx in rad_profile_dict['slit_profile_dict']['list_angle_idx']:

                        rad_arcsec = helper_func.CoordTools.transform_pix2world_scale(
                            length_in_pix=rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
                            wcs=obs_cutout_dict['%s_img_cutout' % band].wcs)

                        if not neg_flx_in_apert:
                            if best_gauss_dict['mask_good_fits'][idx]:
                                ax_slit_profiles.plot(rad_arcsec,
                                                      rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
                                                      color='k', linewidth=3)
                            else:
                                ax_slit_profiles.plot(rad_arcsec,
                                                      rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
                                                      color='gray', linewidth=1)
                        else:
                            ax_slit_profiles.plot(rad_arcsec,
                                                  rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
                                                  color='gray', linewidth=1)
                    if not neg_flx_in_apert:
                        ax_slit_profiles.plot(
                            [phot_aperture_arcsec_list[band_idx], phot_aperture_arcsec_list[band_idx]],
                            [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:red', linewidth=4)
                        ax_slit_profiles.plot(
                            [-phot_aperture_arcsec_list[band_idx], -phot_aperture_arcsec_list[band_idx]],
                            [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:red', linewidth=4)
                        ax_slit_profiles.plot([bkg_rad_in_arcsec_list[band_idx], bkg_rad_in_arcsec_list[band_idx]],
                                              [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:blue', linestyle='--',
                                              linewidth=4)
                        ax_slit_profiles.plot([-bkg_rad_in_arcsec_list[band_idx], -bkg_rad_in_arcsec_list[band_idx]],
                                              [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:blue', linestyle='--',
                                              linewidth=4)
                        ax_slit_profiles.plot([bkg_rad_out_arcsec_list[band_idx], bkg_rad_out_arcsec_list[band_idx]],
                                              [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:blue', linestyle='--',
                                              linewidth=4)
                        ax_slit_profiles.plot([-bkg_rad_out_arcsec_list[band_idx], -bkg_rad_out_arcsec_list[band_idx]],
                                              [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:blue', linestyle='--',
                                              linewidth=4)

                    if not neg_flx_in_apert:
                        ax_slit_profiles.plot(dummy_rad_arcsec, mean_gauss, linewidth=5, color='tab:red')

                        plotting_tools.AxisTools.frame2axis(ax=ax_slit_profiles, color='k', line_width=3)
                        ax_slit_profiles.tick_params(which='both', width=3, length=5, direction='in', color='k',
                                                     labelsize=fontsize_large_label)

                        plotting_tools.StrTools.display_text_in_corner(
                            ax=ax_slit_profiles, text=r'F$_{corr.}$=' + '\n %.2f' % corr_fact,
                            x_frac=0.98, y_frac=0.98, horizontal_alignment='right',
                            vertical_alignment='top',
                            fontsize=fontsize_large_label, text_color='k')

                    else:
                        plotting_tools.StrTools.display_text_in_corner(
                            ax=ax_slit_profiles, text=r'No Signal',
                            x_frac=0.98, y_frac=0.98, horizontal_alignment='right',
                            vertical_alignment='top',
                            fontsize=fontsize_large_label, text_color='k')

                    ax_slit_profiles.set_xticklabels([])
                    ax_slit_profiles.set_yticklabels([])

                    idx_col += 1
                    if idx_col >= n_cols:
                        idx_col = 0
                        idx_row += 3

                    # plot sed point
                    mean_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='mean_wave', unit='mu')
                    min_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='min_wave', unit='mu')
                    max_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='max_wave', unit='mu')

                    # plot SED point only when there is a detection

                    if (not neg_flx_in_apert) & (
                            (forced_photomerty_dict['src_flux'] / forced_photomerty_dict['src_flux_err']) > 3):

                        ax_sed.errorbar(mean_band_wavelength, forced_photomerty_dict['src_flux'],
                                        xerr=[[mean_band_wavelength - min_band_wavelength],
                                              [max_band_wavelength - mean_band_wavelength]],
                                        yerr=forced_photomerty_dict['src_flux_err'],
                                        fmt='.', color='k', ms=60)

                        ax_sed.errorbar(mean_band_wavelength, forced_photomerty_dict['src_flux'] * corr_fact,
                                        xerr=[[mean_band_wavelength - min_band_wavelength],
                                              [max_band_wavelength - mean_band_wavelength]],
                                        yerr=forced_photomerty_dict['src_flux_err'] * corr_fact,
                                        fmt='.', color='tab:red', ms=45)

                        ax_sed.errorbar(mean_band_wavelength,
                                        forced_photomerty_dict['src_flux'] * corr_fact * gal_red_corr_fact,
                                        xerr=[[mean_band_wavelength - min_band_wavelength],
                                              [max_band_wavelength - mean_band_wavelength]],
                                        yerr=forced_photomerty_dict['src_flux_err'] * corr_fact * gal_red_corr_fact,
                                        fmt='.', color='tab:blue', ms=30)
                    else:
                        ax_sed.errorbar(mean_band_wavelength, forced_photomerty_dict['src_flux_err'],
                                        uplims=True,
                                        xerr=[[mean_band_wavelength - min_band_wavelength],
                                              [max_band_wavelength - mean_band_wavelength]],
                                        yerr=forced_photomerty_dict['src_flux_err'],
                                        fmt='.', color='k', ms=60)

                # fill table with values
                # correct for flux
                if not neg_flx_in_apert:
                    flux_table['%s_flux' % band][running_idx] = forced_photomerty_dict[
                                                                    'src_flux'] * corr_fact * gal_red_corr_fact
                    flux_table['%s_flux_err' % band][running_idx] = forced_photomerty_dict[
                                                                        'src_flux_err'] * corr_fact * gal_red_corr_fact
                else:
                    flux_table['%s_flux' % band][running_idx] = forced_photomerty_dict['src_flux_err']
                    flux_table['%s_flux_err' % band][running_idx] = forced_photomerty_dict['src_flux_err']
                # the aperture corr factors
                flux_table['%s_apert_corr_fact' % band][running_idx] = corr_fact
                flux_table['%s_apert_gal_ext_fact' % band][running_idx] = gal_red_corr_fact
                # uncorrected fluxes
                flux_table['%s_flux_bkg_sub_no_corr' % band][running_idx] = forced_photomerty_dict['src_flux']
                flux_table['%s_flux_apert' % band][running_idx] = forced_photomerty_dict['apert_flux']
                # statistics from aperture and annulus
                flux_table['%s_peak' % band][running_idx] = forced_photomerty_dict['apert_max']
                flux_table['%s_flux_flag' % band][running_idx] = forced_photomerty_dict['flux_measure_ok_flag']
                flux_table['%s_median_bkg' % band][running_idx] = forced_photomerty_dict['bkg_median']
                flux_table['%s_mean_bkg' % band][running_idx] = forced_photomerty_dict['bkg_mean']
                flux_table['%s_bkg_flag' % band][running_idx] = forced_photomerty_dict['bkg_ok_flag']

            if plot_flag:
                ax_sed.set_xscale('log')
                ax_sed.set_yscale('log')
                if obj_name is not None:
                    plotting_tools.StrTools.display_text_in_corner(y_frac=0.95,
                                                                   ax=ax_sed, text=obj_name.upper(),
                                                                   fontsize=fontsize_large_label + 10,
                                                                   text_color='dimgray', path_eff_color='k', )

                plotting_tools.AxisTools.frame2axis(ax=ax_sed, color='k', line_width=3)

                ax_sed.scatter([], [], s=1000, color='k', label='Measured Flux')
                ax_sed.scatter([], [], s=700, color='tab:red', label='Measured Flux + Apert. Corr.')
                ax_sed.scatter([], [], s=400, color='tab:blue', label='Measured Flux + Apert. Corr. + Gal. Ext. Corr.')

                ax_sed.legend(frameon=False, loc=4, fontsize=fontsize_large_label)

                ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fontsize_large_label, labelpad=-10)
                ax_sed.set_ylabel(r'flux [mJy]', fontsize=fontsize_large_label)
                ax_sed.tick_params(axis='both', which='major', width=5, length=15, direction='in', color='k',
                                   labelsize=fontsize_large_label)
                ax_sed.tick_params(axis='both', which='minor', width=3, length=10, direction='in', color='k',
                                   labelsize=fontsize_large_label)

                # plt.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.05)
                if not os.path.isdir(plot_output_path):
                    os.makedirs(plot_output_path)
                fig.savefig(plot_output_path + 'compact_src_photmetry_%s_%i.png' % (self.phot_target_name, obj_idx))
                if save_as_pdf:
                    fig.savefig(plot_output_path + 'compact_src_photmetry_%s_%i.pdf' % (self.phot_target_name, obj_idx))
                plt.close(fig)

        return flux_table







    def compute_compact_dynamic_apert_corr_photometry(
            self, ra_list, dec_list, roi_arcsec, bkg_roi_rad_in_arcsec, bkg_roi_rad_out_arcsec,
            idx_list=None,
            band_list=None,
            include_hst=True, include_nircam=True, include_miri=True, include_astrosat=False,

            profile_fit_rad_frac=1,
            cutout_size=None,
            obj_name=None,
            flux_unit='mJy',
            re_centering=True,
            verbose_flag=True,
            plot_flag=True,
            plot_output_path='plot_output/',
            save_as_pdf=False
    ):
        """
        This function is a first attempt in measuring an aperture corrected photometry of compact sources.

        """

        ########################
        #### organize input ####
        ########################
        # check if the coordinates are a list or just a float
        if isinstance(ra_list, float):
            ra_list = [ra_list]
            dec_list = [dec_list]
        # get an index for sources or make them up
        if idx_list is not None:
            if not isinstance(idx_list, list):
                idx_list = [idx_list]
        else:
            idx_list = np.arange(len(ra_list))

        # get band list
        band_list, hst_band_list, nircam_band_list, miri_band_list, astrosat_band_list = (
            ObsTools.split_obs_band_list(
                band_list=band_list, hst_target_name=self.phot_hst_target_name,
                nircam_target_name=self.phot_nircam_target_name, miri_target_name=self.phot_miri_target_name,
                astrosat_target_name=self.phot_astrosat_target_name,
                            nircam_data_ver=self.nircam_data_ver, miri_data_ver=self.miri_data_ver,
                astrosat_data_ver=self.astrosat_data_ver,
                            include_hst=include_hst, include_nircam=include_nircam, include_miri=include_miri, include_astrosat=include_astrosat))

        # check if the bands are loaded!
        self.load_phangs_bands(band_list=band_list, flux_unit=flux_unit, load_err=True)

        # get observation abd instrument list
        obs_list, instrument_list, target_name_list = (
            ObsTools.get_obs_list(band_list=band_list, hst_band_list=hst_band_list, nircam_band_list=nircam_band_list,
                                  miri_band_list=miri_band_list, astrosat_band_list=astrosat_band_list,
                                  hst_target_name=self.phot_hst_target_name,
                                  nircam_target_name=self.phot_nircam_target_name,
                                  miri_target_name=self.phot_miri_target_name,
                                  astrosat_target_name=self.phot_astrosat_target_name,))

        # get all the aperture and background sizes needed
        (standard_phot_aperture_arcsec_list, phot_aperture_pix_list, phot_aperture_arcsec_list,
         bkg_rad_in_arcsec_list, bkg_rad_out_arcsec_list, fwhm_pix_list, std_pix_list) = (
            self.get_aperture_radii_and_scales(band_list=band_list, instrument_list=instrument_list, obs_list=obs_list,
                                               roi_arcsec=roi_arcsec, bkg_roi_rad_in_arcsec=bkg_roi_rad_in_arcsec,
                                               bkg_roi_rad_out_arcsec=bkg_roi_rad_out_arcsec))

        # prepare flux table columns
        flux_table_names = []
        for band in band_list:
            flux_table_names.append('%s_flux' % band)
            flux_table_names.append('%s_flux_err' % band)
            flux_table_names.append('%s_apert_corr_fact' % band)
            flux_table_names.append('%s_apert_gal_ext_fact' % band)
            flux_table_names.append('%s_flux_bkg_sub_no_corr' % band)
            flux_table_names.append('%s_flux_apert' % band)
            flux_table_names.append('%s_peak' % band)
            flux_table_names.append('%s_flux_flag' % band)
            flux_table_names.append('%s_median_bkg' % band)
            flux_table_names.append('%s_mean_bkg' % band)
            flux_table_names.append('%s_bkg_flag' % band)
        # construct a table
        r_rows = len(idx_list)
        n_cols = len(band_list) * 11
        flux_table = Table(np.zeros((r_rows, n_cols)), names=flux_table_names)

        #################################
        # now loop over all coordinates #
        #################################
        for running_idx, obj_idx, ra, dec in zip(range(len(idx_list)), idx_list, ra_list, dec_list):
            if verbose_flag: print(running_idx, obj_idx, ra, dec)

            if plot_flag:
                fontsize_small_label = 40
                fontsize_large_label = 50

                fig_size_individual = (5, 5)
                n_cols = 7
                n_rows = int(np.ceil((len(band_list)) / n_cols) * 3 + 2)

                fig = plt.figure(figsize=(fig_size_individual[0] * n_cols, fig_size_individual[1] * n_rows))
                gs = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.01, bottom=0.01, right=0.99, top=0.99,
                                      wspace=0.01, hspace=0.01)
                gs_sed = fig.add_gridspec(ncols=n_cols, nrows=n_rows, left=0.08, bottom=0.01, right=0.99, top=0.99,
                                          wspace=0.5, hspace=0.5)
                ax_sed = fig.add_subplot(gs_sed[:2, :])

            ###########################
            # now loop over all bands #
            ###########################
            idx_row = 2
            idx_col = 0
            for band_idx, band in enumerate(band_list):

                if verbose_flag:
                    print(band)

                obs_cutout_dict, ra_re_center, dec_re_center = (
                    self.get_src_cutout_and_recenter(ra=ra, dec=dec, band=band,
                                                     instrument=instrument_list[band_idx], roi_arcsec=roi_arcsec,
                                                     max_rad_arcsec=bkg_rad_out_arcsec_list[band_idx], cutout_size=cutout_size,
                                    re_centering=re_centering))

                # perform photometry
                forced_photomerty_dict = phot_tools.ApertTools.compute_apert_photometry(
                    apert_rad_arcsec=phot_aperture_arcsec_list[band_idx],
                    bkg_rad_annulus_in_arcsec=bkg_rad_in_arcsec_list[band_idx],
                    bkg_rad_annulus_out_arcsec=bkg_rad_out_arcsec_list[band_idx],
                    data=obs_cutout_dict['%s_img_cutout' % band].data,
                    data_err=obs_cutout_dict['%s_err_cutout' % band].data,
                    wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                    ra=ra_re_center, dec=dec_re_center,
                    mask=None, sigma_clip_sig=3, sigma_clip_maxiters=10, sum_method='exact')

                # if the flux inside the aperture is negative there is no way we can do anything!.
                # This is an absolute non detection!
                if (forced_photomerty_dict['apert_flux'] <= 0) | (forced_photomerty_dict['src_flux'] <= 0):
                    neg_flx_in_apert = True
                else:
                    neg_flx_in_apert = False

                ########################################
                #### Perform an aperture correction ####
                ########################################
                # radial profile dict with also slit profiles
                rad_profile_dict = phot_tools.ProfileTools.get_rad_profile_dict(
                    img=obs_cutout_dict['%s_img_cutout' % band].data - forced_photomerty_dict['bkg_median'],
                    wcs=obs_cutout_dict['%s_img_cutout' % band].wcs,
                    ra=ra_re_center, dec=dec_re_center,
                    n_slits=12, max_rad_arcsec=bkg_rad_out_arcsec_list[band_idx] * 1.1,
                    img_err=obs_cutout_dict['%s_err_cutout' % band].data,
                    img_mask=np.isnan(obs_cutout_dict['%s_img_cutout' % band].data -
                                      forced_photomerty_dict['bkg_median']))

                # only if there is a signal in the aperture
                if not neg_flx_in_apert:

                    rad_of_interest_pix = helper_func.CoordTools.transform_world2pix_scale(
                        length_in_arcsec=phot_aperture_arcsec_list[band_idx],
                        wcs=obs_cutout_dict['%s_img_cutout' % band].wcs)

                    amp_list, mu_list, sig_list, amp_err_list, mu_err_list, sig_err_list = (
                        phot_tools.ProfileTools.fit_gauss2rad_profiles(
                            rad_profile_dict=rad_profile_dict['slit_profile_dict'], std_pix=std_pix_list[band_idx],
                            upper_sig_fact=10, central_rad_fact=5,
                            radius_of_interest=rad_of_interest_pix * profile_fit_rad_frac))

                    # now select the best gaussian fit
                    best_gauss_dict = phot_tools.ProfileTools.get_best_gaussian_profile_params(
                        amp_list=amp_list, mu_list=mu_list, sig_list=sig_list,
                        peak_acceptance_rad_pix=std_pix_list[band_idx] * 1, n_good_fits_needed=4)

                    # find the best gaussian fit!
                    mean_mu_arcsec = helper_func.CoordTools.transform_pix2world_scale(
                        length_in_pix=best_gauss_dict['mean_mu'], wcs=obs_cutout_dict['%s_img_cutout' % band].wcs)
                    mean_sig_arcsec = helper_func.CoordTools.transform_pix2world_scale(
                        length_in_pix=best_gauss_dict['mean_sig'], wcs=obs_cutout_dict['%s_img_cutout' % band].wcs)

                    # get a dummy gaussian
                    n_pixels_in_bkg_rad = int(helper_func.CoordTools.transform_world2pix_scale(
                        length_in_arcsec=bkg_rad_out_arcsec_list[band_idx],
                        wcs=obs_cutout_dict['%s_img_cutout' % band].wcs) * 2)
                    dummy_rad_arcsec = np.linspace(-bkg_rad_out_arcsec_list[band_idx],
                                                   bkg_rad_out_arcsec_list[band_idx],
                                                   n_pixels_in_bkg_rad * 10)
                    mean_gauss = helper_func.FitTools.gaussian_func(
                        amp=best_gauss_dict['mean_amp'], mu=mean_mu_arcsec,
                        sig=mean_sig_arcsec, x_data=dummy_rad_arcsec)

                    print('mean_sig_arcsec ', mean_sig_arcsec)
                    # correction factor
                    corr_fact, too_extended_flag = phot_tools.PSFTools.get_apert_gauss_corr_fact(
                        band=band, instrument=instrument_list[band_idx], apert_rad=phot_aperture_arcsec_list[band_idx],
                        std=mean_sig_arcsec)

                else:
                    corr_fact = None

                # get galactic reddening correction
                fore_ground_ext = DustTools.get_target_gal_ext_band(
                    target=self.phot_target_name, obs=obs_list[band_idx], band=band, ext_law='G23')

                gal_red_corr_fact = 10 ** (fore_ground_ext / 2.5)

                if plot_flag:

                    ######################
                    ###### plot img ######
                    ######################
                    # add axis
                    ax_img = fig.add_subplot(gs[idx_row, idx_col],
                                             projection=obs_cutout_dict['%s_img_cutout' % band].wcs)
                    # get an image norm
                    mean, median, std = sigma_clipped_stats(obs_cutout_dict['%s_img_cutout' % band].data, sigma=3.0)
                    vmin = median - 1 * std
                    vmax = median + 30 * std
                    norm_img = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
                    # display image
                    ax_img.imshow(obs_cutout_dict['%s_img_cutout' % band].data, norm=norm_img, cmap='Greys')
                    # cosmetics
                    ax_img.axis('off')
                    plotting_tools.StrTools.display_text_in_corner(
                        ax=ax_img, text=band.upper(), fontsize=fontsize_large_label, text_color='tab:red',
                        path_eff_color='k', )
                    coords_world = SkyCoord(ra=ra_re_center * u.deg, dec=dec_re_center * u.deg)
                    # add a scale bar
                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_img, img_shape=obs_cutout_dict['%s_img_cutout' % band].data.shape,
                        wcs=obs_cutout_dict['%s_img_cutout' % band].wcs, bar_length=1, length_unit='arcsec',
                        bar_color='red', text_color='red', line_width=4, fontsize=fontsize_small_label,
                        va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01,
                        path_eff=True, path_eff_color='k')
                    # plot aperture_radius and background annuli
                    plotting_tools.WCSPlottingTools.plot_coord_circle(
                        ax=ax_img, pos=coords_world, rad=phot_aperture_arcsec_list[band_idx],
                        color='tab:red', line_style='-', line_width=5, alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(
                        ax=ax_img, pos=coords_world, rad=bkg_rad_in_arcsec_list[band_idx],
                        color='tab:blue', line_style='--', line_width=5, alpha=1., fill=False)
                    plotting_tools.WCSPlottingTools.plot_coord_circle(
                        ax=ax_img, pos=coords_world, rad=bkg_rad_out_arcsec_list[band_idx],
                        color='tab:blue', line_style='--', line_width=5, alpha=1., fill=False)

                    ##################################
                    ###### plot radial profiles ######
                    ##################################
                    # add axis
                    ax_profile = fig.add_subplot(gs[idx_row + 1, idx_col])

                    ax_profile.plot(rad_profile_dict['rad'], rad_profile_dict['profile'], linewidth=4, color='k')
                    ax_profile.fill_between(x=rad_profile_dict['rad'],
                                            y1=rad_profile_dict['profile'] + rad_profile_dict['profile_err'],
                                            y2=rad_profile_dict['profile'] - rad_profile_dict['profile_err'],
                                            color='lightgray')
                    max_rad_pofile = np.max(rad_profile_dict['profile'])

                    ax_profile.plot([phot_aperture_arcsec_list[band_idx], phot_aperture_arcsec_list[band_idx]],
                                    [0, max_rad_pofile], color='tab:red', linewidth=4)
                    ax_profile.plot([bkg_rad_in_arcsec_list[band_idx], bkg_rad_in_arcsec_list[band_idx]],
                                    [0, max_rad_pofile], color='tab:blue', linestyle='--',
                                    linewidth=4)
                    ax_profile.plot([bkg_rad_out_arcsec_list[band_idx], bkg_rad_out_arcsec_list[band_idx]],
                                    [0, max_rad_pofile], color='tab:blue', linestyle='--',
                                    linewidth=4)

                    ax_profile.set_xticklabels([])
                    ax_profile.set_yticklabels([])
                    plotting_tools.AxisTools.frame2axis(ax=ax_profile, color='k', line_width=3)
                    ax_profile.tick_params(which='both', width=3, length=5, direction='in', color='k',
                                           labelsize=fontsize_large_label)

                    ax_slit_profiles = fig.add_subplot(gs[idx_row + 2, idx_col])
                    # plot all the gaussians
                    for idx in rad_profile_dict['slit_profile_dict']['list_angle_idx']:

                        rad_arcsec = helper_func.CoordTools.transform_pix2world_scale(
                            length_in_pix=rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
                            wcs=obs_cutout_dict['%s_img_cutout' % band].wcs)

                        if not neg_flx_in_apert:
                            if best_gauss_dict['mask_good_fits'][idx]:
                                ax_slit_profiles.plot(rad_arcsec,
                                                      rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
                                                      color='k', linewidth=3)
                            else:
                                ax_slit_profiles.plot(rad_arcsec,
                                                      rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
                                                      color='gray', linewidth=1)
                        else:
                            ax_slit_profiles.plot(rad_arcsec,
                                                  rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
                                                  color='gray', linewidth=1)
                    if not neg_flx_in_apert:
                        ax_slit_profiles.plot(
                            [phot_aperture_arcsec_list[band_idx], phot_aperture_arcsec_list[band_idx]],
                            [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:red', linewidth=4)
                        ax_slit_profiles.plot(
                            [-phot_aperture_arcsec_list[band_idx], -phot_aperture_arcsec_list[band_idx]],
                            [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:red', linewidth=4)
                        ax_slit_profiles.plot([bkg_rad_in_arcsec_list[band_idx], bkg_rad_in_arcsec_list[band_idx]],
                                              [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:blue', linestyle='--',
                                              linewidth=4)
                        ax_slit_profiles.plot([-bkg_rad_in_arcsec_list[band_idx], -bkg_rad_in_arcsec_list[band_idx]],
                                              [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:blue', linestyle='--',
                                              linewidth=4)
                        ax_slit_profiles.plot([bkg_rad_out_arcsec_list[band_idx], bkg_rad_out_arcsec_list[band_idx]],
                                              [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:blue', linestyle='--',
                                              linewidth=4)
                        ax_slit_profiles.plot([-bkg_rad_out_arcsec_list[band_idx], -bkg_rad_out_arcsec_list[band_idx]],
                                              [0, best_gauss_dict['mean_amp'] * 0.5], color='tab:blue', linestyle='--',
                                              linewidth=4)

                    if not neg_flx_in_apert:
                        ax_slit_profiles.plot(dummy_rad_arcsec, mean_gauss, linewidth=5, color='tab:red')

                        plotting_tools.AxisTools.frame2axis(ax=ax_slit_profiles, color='k', line_width=3)
                        ax_slit_profiles.tick_params(which='both', width=3, length=5, direction='in', color='k',
                                                     labelsize=fontsize_large_label)

                        plotting_tools.StrTools.display_text_in_corner(
                            ax=ax_slit_profiles, text=r'F$_{corr.}$=' + '\n %.2f' % corr_fact,
                            x_frac=0.98, y_frac=0.98, horizontal_alignment='right',
                            vertical_alignment='top',
                            fontsize=fontsize_large_label, text_color='k')

                    else:
                        plotting_tools.StrTools.display_text_in_corner(
                            ax=ax_slit_profiles, text=r'No Signal',
                            x_frac=0.98, y_frac=0.98, horizontal_alignment='right',
                            vertical_alignment='top',
                            fontsize=fontsize_large_label, text_color='k')

                    ax_slit_profiles.set_xticklabels([])
                    ax_slit_profiles.set_yticklabels([])

                    idx_col += 1
                    if idx_col >= n_cols:
                        idx_col = 0
                        idx_row += 3

                    # plot sed point
                    mean_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='mean_wave', unit='mu')
                    min_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='min_wave', unit='mu')
                    max_band_wavelength = ObsTools.get_phangs_telescope_wave(
                        target=target_name_list[band_idx], band=band, telescope=obs_list[band_idx],
                        wave_estimator='max_wave', unit='mu')

                    # plot SED point only when there is a detection

                    if (not neg_flx_in_apert) & (
                            (forced_photomerty_dict['src_flux'] / forced_photomerty_dict['src_flux_err']) > 3):

                        ax_sed.errorbar(mean_band_wavelength, forced_photomerty_dict['src_flux'],
                                        xerr=[[mean_band_wavelength - min_band_wavelength],
                                              [max_band_wavelength - mean_band_wavelength]],
                                        yerr=forced_photomerty_dict['src_flux_err'],
                                        fmt='.', color='k', ms=60)

                        ax_sed.errorbar(mean_band_wavelength, forced_photomerty_dict['src_flux'] * corr_fact,
                                        xerr=[[mean_band_wavelength - min_band_wavelength],
                                              [max_band_wavelength - mean_band_wavelength]],
                                        yerr=forced_photomerty_dict['src_flux_err'] * corr_fact,
                                        fmt='.', color='tab:red', ms=45)

                        ax_sed.errorbar(mean_band_wavelength,
                                        forced_photomerty_dict['src_flux'] * corr_fact * gal_red_corr_fact,
                                        xerr=[[mean_band_wavelength - min_band_wavelength],
                                              [max_band_wavelength - mean_band_wavelength]],
                                        yerr=forced_photomerty_dict['src_flux_err'] * corr_fact * gal_red_corr_fact,
                                        fmt='.', color='tab:blue', ms=30)
                    else:
                        ax_sed.errorbar(mean_band_wavelength, forced_photomerty_dict['src_flux_err'],
                                        uplims=True,
                                        xerr=[[mean_band_wavelength - min_band_wavelength],
                                              [max_band_wavelength - mean_band_wavelength]],
                                        yerr=forced_photomerty_dict['src_flux_err'],
                                        fmt='.', color='k', ms=60)

                # fill table with values
                # correct for flux
                if not neg_flx_in_apert:
                    flux_table['%s_flux' % band][running_idx] = forced_photomerty_dict[
                                                                    'src_flux'] * corr_fact * gal_red_corr_fact
                    flux_table['%s_flux_err' % band][running_idx] = forced_photomerty_dict[
                                                                        'src_flux_err'] * corr_fact * gal_red_corr_fact
                else:
                    flux_table['%s_flux' % band][running_idx] = forced_photomerty_dict['src_flux_err']
                    flux_table['%s_flux_err' % band][running_idx] = forced_photomerty_dict['src_flux_err']
                # the aperture corr factors
                flux_table['%s_apert_corr_fact' % band][running_idx] = corr_fact
                flux_table['%s_apert_gal_ext_fact' % band][running_idx] = gal_red_corr_fact
                # uncorrected fluxes
                flux_table['%s_flux_bkg_sub_no_corr' % band][running_idx] = forced_photomerty_dict['src_flux']
                flux_table['%s_flux_apert' % band][running_idx] = forced_photomerty_dict['apert_flux']
                # statistics from aperture and annulus
                flux_table['%s_peak' % band][running_idx] = forced_photomerty_dict['apert_max']
                flux_table['%s_flux_flag' % band][running_idx] = forced_photomerty_dict['flux_measure_ok_flag']
                flux_table['%s_median_bkg' % band][running_idx] = forced_photomerty_dict['bkg_median']
                flux_table['%s_mean_bkg' % band][running_idx] = forced_photomerty_dict['bkg_mean']
                flux_table['%s_bkg_flag' % band][running_idx] = forced_photomerty_dict['bkg_ok_flag']

            if plot_flag:
                ax_sed.set_xscale('log')
                ax_sed.set_yscale('log')
                if obj_name is not None:
                    plotting_tools.StrTools.display_text_in_corner(y_frac=0.95,
                                                                   ax=ax_sed, text=obj_name.upper(),
                                                                   fontsize=fontsize_large_label + 10,
                                                                   text_color='dimgray', path_eff_color='k', )

                plotting_tools.AxisTools.frame2axis(ax=ax_sed, color='k', line_width=3)

                ax_sed.scatter([], [], s=1000, color='k', label='Measured Flux')
                ax_sed.scatter([], [], s=700, color='tab:red', label='Measured Flux + Apert. Corr.')
                ax_sed.scatter([], [], s=400, color='tab:blue', label='Measured Flux + Apert. Corr. + Gal. Ext. Corr.')

                ax_sed.legend(frameon=False, loc=4, fontsize=fontsize_large_label)

                ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fontsize_large_label, labelpad=-10)
                ax_sed.set_ylabel(r'flux [mJy]', fontsize=fontsize_large_label)
                ax_sed.tick_params(axis='both', which='major', width=5, length=15, direction='in', color='k',
                                   labelsize=fontsize_large_label)
                ax_sed.tick_params(axis='both', which='minor', width=3, length=10, direction='in', color='k',
                                   labelsize=fontsize_large_label)

                # plt.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.99, wspace=0.01, hspace=0.05)
                if not os.path.isdir(plot_output_path):
                    os.makedirs(plot_output_path)
                fig.savefig(plot_output_path + 'compact_src_photmetry_%s_%i.png' % (self.phot_target_name, obj_idx))
                if save_as_pdf:
                    fig.savefig(plot_output_path + 'compact_src_photmetry_%s_%i.pdf' % (self.phot_target_name, obj_idx))
                plt.close(fig)

        return flux_table
























# old code


    @staticmethod
    def measure_morph_photometry(rad_profile_dict, psf_dict, img, bkg, img_err, wcs, ra, dec):

        # get average value in the PSF aperture:
        # print(psf_dict['gaussian_fwhm'])
        # print(psf_dict['gaussian_std'])

        radius_of_interes = psf_dict['gaussian_std'] * 3

        central_apert_stats_source = phot_tools.PhotTools.get_circ_apert_stats(data=img - bkg, data_err=img_err,
                                                                               wcs=wcs,
                                                                               ra=ra, dec=dec,
                                                                               aperture_rad=radius_of_interes)
        central_apert_stats_bkg = phot_tools.PhotTools.get_circ_apert_stats(data=bkg, data_err=img_err, wcs=wcs,
                                                                            ra=ra, dec=dec,
                                                                            aperture_rad=radius_of_interes)

        amp_list = []
        mu_list = []
        sig_list = []
        amp_err_list = []
        mu_err_list = []
        sig_err_list = []

        # fig, ax = plt.subplots(nrows=len(rad_profile_dict['slit_profile_dict']['list_angle_idx']) +1)

        for idx in rad_profile_dict['slit_profile_dict']['list_angle_idx']:
            mask_center = ((rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] > psf_dict[
                'gaussian_std'] * 3 * -1) &
                           (rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] < psf_dict[
                               'gaussian_std'] * 3))
            min_value_in_center = np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_center])
            max_value_in_center = np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_center])

            # ax[idx].plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            # ax[-1].plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            #
            # ax[idx].errorbar(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #          rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              yerr=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_err'],
            #          fmt='.')
            # plt.plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')
            # plt.show()

            mask_central_pixels = ((rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] >
                                    psf_dict['gaussian_std'] * -3) &
                                   (rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'] <
                                    psf_dict['gaussian_std'] * 3))
            # ax[idx].scatter(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'][mask_central_pixels],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_central_pixels],
            #              color='red')

            # select the lower amplitude. There is a chance that the values are negative
            lower_amp = min_value_in_center
            upper_amp = max_value_in_center + np.abs(max_value_in_center * 2)

            # plt.plot(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'],
            #              rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'],
            #              color='gray')

            gaussian_fit_dict = helper_func.FitTools.fit_gauss(
                x_data=rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data'][mask_central_pixels],
                y_data=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_data'][mask_central_pixels],
                y_data_err=rad_profile_dict['slit_profile_dict'][str(idx)]['profile_err'][mask_central_pixels],
                amp_guess=max_value_in_center, mu_guess=0, sig_guess=psf_dict['gaussian_std'],
                lower_amp=lower_amp, upper_amp=upper_amp,
                lower_mu=psf_dict['gaussian_std'] * -5, upper_mu=psf_dict['gaussian_std'] * 5,
                lower_sigma=psf_dict['gaussian_std'], upper_sigma=psf_dict['gaussian_std'] * 5)

            # dummy_rad = np.linspace(np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']),
            #                         np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']), 500)
            # gauss = helper_func.FitTools.gaussian_func(
            #     amp=gaussian_fit_dict['amp'], mu=gaussian_fit_dict['mu'], sig=gaussian_fit_dict['sig'], x_data=dummy_rad)
            #
            # # ax[idx].plot(dummy_rad, gauss)
            # plt.plot(dummy_rad, gauss)
            # plt.show()

            amp_list.append(gaussian_fit_dict['amp'])
            mu_list.append(gaussian_fit_dict['mu'])
            sig_list.append(gaussian_fit_dict['sig'])

            amp_err_list.append(gaussian_fit_dict['amp_err'])
            mu_err_list.append(gaussian_fit_dict['mu_err'])
            sig_err_list.append(gaussian_fit_dict['sig_err'])

        # get the best matching gauss

        amp_list = np.array(amp_list)
        mu_list = np.array(mu_list)
        sig_list = np.array(sig_list)
        amp_err_list = np.array(amp_err_list)
        mu_err_list = np.array(mu_err_list)
        sig_err_list = np.array(sig_err_list)

        # get all the gaussian functions that are central
        mask_mu = np.abs(mu_list) < psf_dict['gaussian_std'] * 1
        mask_amp = (amp_list > 0)

        # if no function was detected in the center
        if sum(mask_mu * mask_amp) == 0:
            # non detection
            mean_amp = central_apert_stats_source.max
            mean_mu = 0
            mean_sig = psf_dict['gaussian_std']
            # get flux inside the 3 sigma aperture
            flux = central_apert_stats_source.sum
            flux_err = np.sqrt(central_apert_stats_source.sum_err ** 2 + central_apert_stats_bkg.sum_err ** 2)
            detect_flag = False
        else:
            # print(sum(mask_mu * mask_amp))
            # print(amp_list[mask_mu * mask_amp])
            # print(mu_list[mask_mu * mask_amp])
            # print(sig_list[mask_mu * mask_amp])

            mean_amp = np.mean(amp_list[mask_mu * mask_amp])
            mean_mu = np.mean(mu_list[mask_mu * mask_amp])
            mean_sig = np.mean(sig_list[mask_mu * mask_amp])

            mean_amp_err = np.mean(amp_err_list[mask_mu * mask_amp])
            mean_sig_err = np.mean(sig_err_list[mask_mu * mask_amp])

            # we need to do the sigma in pixel scale though
            mean_sig_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=mean_sig, wcs=wcs)
            # get gaussian integaral as flux
            flux = mean_amp * 2 * np.pi * (mean_sig_pix ** 2)
            flux_err = np.sqrt(mean_amp_err ** 2 * (2 * mean_sig_err) ** 2)
            flux_err = np.sqrt(flux_err ** 2 + central_apert_stats_bkg.sum_err ** 2)
            detect_flag = True

        dummy_rad = np.linspace(np.min(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']),
                                np.max(rad_profile_dict['slit_profile_dict'][str(idx)]['radius_data']), 500)
        gauss = helper_func.FitTools.gaussian_func(
            amp=mean_amp, mu=mean_mu, sig=mean_sig, x_data=dummy_rad)

        photometry_dict = {
            'dummy_rad': dummy_rad,
            'gauss': gauss,
            'flux': flux,
            'flux_err': flux_err,
            'detect_flag': detect_flag
        }
        return photometry_dict

        # ax[-1].plot(dummy_rad, gauss)
        #
        #
        # plt.show()
        # # exit()



# old code


