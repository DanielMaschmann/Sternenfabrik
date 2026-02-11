"""
Tool to visualize PHANGS imaging data
"""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from regions import EllipseSkyRegion
from werkzeugkiste import helper_func, phys_params, phot_tools
from malkasten import plotting_tools
from obszugang import ObsTools, obs_info, ClusterCatAccess
from sternenfabrik import plot_params
from sternenfabrik.phot_lab import PhotLab


class PlotFabrik(PhotLab):
    """
    Class to plot cutouts in multiple bands
    """

    def __init__(self, target_name=None, phot_hst_target_name=None, phot_hst_ha_cont_sub_target_name=None,
                 phot_nircam_target_name=None, phot_miri_target_name=None, phot_astrosat_target_name=None,
                 x_target_name=None, radio_target_name=None,
                 nircam_data_ver='v1p1p1', miri_data_ver='v1p1p1', astrosat_data_ver='v1p0',
                 nirspec_data_ver=None, miri_mrs_data_ver=None):
        PhotLab.__init__(self, target_name=target_name, phot_hst_target_name=phot_hst_target_name,
            phot_hst_ha_cont_sub_target_name=phot_hst_ha_cont_sub_target_name,
            phot_nircam_target_name=phot_nircam_target_name, phot_miri_target_name=phot_miri_target_name,
            phot_astrosat_target_name=phot_astrosat_target_name, x_target_name=x_target_name,
                         radio_target_name=radio_target_name,
            nircam_data_ver=nircam_data_ver, miri_data_ver=miri_data_ver, astrosat_data_ver=astrosat_data_ver,
                         nirspec_data_ver=nirspec_data_ver,  miri_mrs_data_ver=miri_mrs_data_ver)

    def plot_hst_overview_panel(self, fig, fig_dict, ra_box=None, dec_box=None):
        """
        """

        # get hst overview image
        band_list = [ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                               filter_name=fig_dict['overview_red_band']),
                     ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                               filter_name=fig_dict['overview_green_band']),
                     ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                               filter_name=fig_dict['overview_blue_band'])]

        # get hst_bvi_zoom_in
        img_overview, wcs_overview = self.get_target_overview_rgb_img(
            red_band=band_list[0], green_band=band_list[1], blue_band=band_list[2],
            overview_img_size=fig_dict['overview_img_pixel_size'], **fig_dict['overview_img_params'])

        # plot the overview image
        # create overview panel axis
        ax_img_overview = fig.add_axes([fig_dict['overview_left_align'], fig_dict['overview_bottom_align'],
                                        fig_dict['overview_width'], fig_dict['overview_height']],
                                       projection=wcs_overview)
        # plot rgb image
        ax_img_overview.imshow(img_overview)
        # add text
        plotting_tools.StrTools.display_text_in_corner(ax=ax_img_overview, text='HST',
                                                       fontsize=fig_dict['overview_title_font_size'],
                                                       text_color='white', x_frac=0.02, y_frac=0.98,
                                                       horizontal_alignment='left', vertical_alignment='top',
                                                       path_eff=True, path_err_linewidth=3, path_eff_color='white')
        plotting_tools.StrTools.display_text_in_corner(ax=ax_img_overview, text=fig_dict['overview_red_band'],
                                                       fontsize=fig_dict['overview_title_font_size'],
                                                       text_color='red', x_frac=0.02, y_frac=0.93,
                                                       horizontal_alignment='left', vertical_alignment='top',
                                                       path_eff=True, path_err_linewidth=3, path_eff_color='white')
        plotting_tools.StrTools.display_text_in_corner(ax=ax_img_overview, text=fig_dict['overview_green_band'],
                                                       fontsize=fig_dict['overview_title_font_size'],
                                                       text_color='green', x_frac=0.02, y_frac=0.88,
                                                       horizontal_alignment='left', vertical_alignment='top',
                                                       path_eff=True, path_err_linewidth=3, path_eff_color='white')
        plotting_tools.StrTools.display_text_in_corner(ax=ax_img_overview, text=fig_dict['overview_blue_band'],
                                                       fontsize=fig_dict['overview_title_font_size'],
                                                       text_color='blue', x_frac=0.02, y_frac=0.83,
                                                       horizontal_alignment='left', vertical_alignment='top',
                                                       path_eff=True, path_err_linewidth=3, path_eff_color='white')

        ax_img_overview.set_title(self.phot_hst_target_name.upper(), fontsize=fig_dict['overview_title_font_size'])
        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_img_overview, ra_tick_label=True, dec_tick_label=True,
                                                        ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                                                        ra_minpad=0.8, dec_minpad=0.8, ra_tick_color='white',
                                                        dec_tick_color='white', ra_label_color='k', dec_label_color='k',
                                                        fontsize=fig_dict['overview_label_size'],
                                                        labelsize=fig_dict['overview_label_size'],
                                                        ra_minor_ticks=True, dec_minor_ticks=True)
        if (ra_box is not None) & (dec_box is not None):
            plotting_tools.WCSPlottingTools.draw_box(ax=ax_img_overview, wcs=wcs_overview,
                                                     coord=SkyCoord(ra=ra_box * u.deg, dec=dec_box * u.deg),
                                                     box_size=fig_dict['env_cutout_size'], color='red', line_style='--')

        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
            ax=ax_img_overview, img_shape=img_overview.shape, wcs=wcs_overview,
            bar_length=fig_dict['overview_scale_bar_length'], length_unit='kpc',
            phangs_target=self.phot_target_name,
            bar_color='white', text_color='white',
            line_width=4, fontsize=fig_dict['overview_label_size'],
            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

    def get_target_overview_rgb_img(self,
                                    red_band, green_band, blue_band,
                                    red_obs='hst', green_obs='hst', blue_obs='hst',
                                    ref_filter='red',
                                    overview_img_size=(500, 500), **rgb_img_kwargs):
        """
        Function to create an overview RGB image of PHANGS HST observations

        Parameters
        ----------
        red_band, green_band, blue_band : str
            Can be specified to any hst band
        ref_filter: str
            Band which is used for the image limits. can be red green or blue
        overview_img_size : tuple
            denotes the shape of the new image

        Returns
        -------
        rgb_image: ``numpy.ndarray``
        wcs: ``astropy.wcs.WCS``
        """

        # band list need to be loaded
        self.load_obs_bands(band_list=[red_band, green_band, blue_band], flux_unit='MJy/sr', load_err=False)
        # get overview image

        non_zero_elements = np.where(getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))[
                                         '%s_data_img' % eval('%s_band' % ref_filter)] != 0)

        min_index_ra_axis_x_val = int(np.mean(non_zero_elements[1][non_zero_elements[1] ==
                                                                   np.min(non_zero_elements[1])]))
        min_index_ra_axis_y_val = int(np.mean(non_zero_elements[0][non_zero_elements[1] ==
                                                                   np.min(non_zero_elements[1])]))
        max_index_ra_axis_x_val = int(np.mean(non_zero_elements[1][non_zero_elements[1] ==
                                                                   np.max(non_zero_elements[1])]))
        max_index_ra_axis_y_val = int(np.mean(non_zero_elements[0][non_zero_elements[1] ==
                                                                   np.max(non_zero_elements[1])]))
        min_index_dec_axis_x_val = int(np.mean(non_zero_elements[1][non_zero_elements[0] ==
                                                                    np.min(non_zero_elements[0])]))
        min_index_dec_axis_y_val = int(np.mean(non_zero_elements[0][non_zero_elements[0] ==
                                                                    np.min(non_zero_elements[0])]))
        max_index_dec_axis_x_val = int(np.mean(non_zero_elements[1][non_zero_elements[0] ==
                                                                    np.max(non_zero_elements[0])]))
        max_index_dec_axis_y_val = int(np.mean(non_zero_elements[0][non_zero_elements[0] ==
                                                                    np.max(non_zero_elements[0])]))

        ra_max = getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))[
            '%s_wcs_img' % eval('%s_band' % ref_filter)].pixel_to_world(
            min_index_ra_axis_x_val, min_index_ra_axis_y_val).ra.value
        ra_min = getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))[
            '%s_wcs_img' % eval('%s_band' % ref_filter)].pixel_to_world(
            max_index_ra_axis_x_val, max_index_ra_axis_y_val).ra.value

        dec_min = getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))[
            '%s_wcs_img' % eval('%s_band' % ref_filter)].pixel_to_world(
            min_index_dec_axis_x_val, min_index_dec_axis_y_val).dec.value
        dec_max = getattr(self, '%s_bands_data' % eval('%s_obs' % ref_filter))[
            '%s_wcs_img' % eval('%s_band' % ref_filter)].pixel_to_world(
            max_index_dec_axis_x_val, max_index_dec_axis_y_val).dec.value

        new_wcs = helper_func.CoordTools.construct_wcs(ra_min=ra_min, ra_max=ra_max, dec_min=dec_max, dec_max=dec_min,
                                                       img_shape=overview_img_size, quadratic_image=True)

        img_data_red = helper_func.CoordTools.reproject_image(
            data=getattr(self, '%s_bands_data' % red_obs)['%s_data_img' % red_band],
            wcs=getattr(self, '%s_bands_data' % red_obs)['%s_wcs_img' % red_band],
            new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_green = helper_func.CoordTools.reproject_image(
            data=getattr(self, '%s_bands_data' % green_obs)['%s_data_img' % green_band],
            wcs=getattr(self, '%s_bands_data' % green_obs)['%s_wcs_img' % green_band],
            new_wcs=new_wcs, new_shape=overview_img_size)
        img_data_blue = helper_func.CoordTools.reproject_image(
            data=getattr(self, '%s_bands_data' % blue_obs)['%s_data_img' % blue_band],
            wcs=getattr(self, '%s_bands_data' % blue_obs)['%s_wcs_img' % blue_band],
            new_wcs=new_wcs, new_shape=overview_img_size)

        img_data_red[img_data_red == 0] = np.nan
        img_data_green[img_data_green == 0] = np.nan
        img_data_blue[img_data_blue == 0] = np.nan

        hst_rgb = plotting_tools.ImgTools.get_rgb_img(data_r=img_data_red, data_g=img_data_green, data_b=img_data_blue,
                                                      **rgb_img_kwargs)

        return hst_rgb, new_wcs

    def get_target_non_quadratic_rgb_img(self,
                                         ra_center, dec_center,
                                         ra_width, dec_width,
                                         red_band, green_band, blue_band,
                                         ref_band_color='red',
                                         pixel_regrade_factor=0.5,
                                         red_obs='hst', green_obs='hst', blue_obs='hst',
                                         **rgb_img_kwargs):

        # band list need to be loaded
        self.load_obs_bands(band_list=[red_band, green_band, blue_band], flux_unit='MJy/sr', load_err=False)

        # get cutouts
        cutout_dict = self.get_band_cutout_dict(
            ra_cutout=ra_center, dec_cutout=dec_center, cutout_size=(dec_width, ra_width),
            band_list=[red_band, green_band, blue_band])

        ref_band = locals()['%s_band' % ref_band_color]
        ref_shape_old = cutout_dict['%s_img_cutout' % ref_band].data.shape
        new_shape = (int(cutout_dict['%s_img_cutout' % ref_band].data.shape[0] * pixel_regrade_factor),
                     int(cutout_dict['%s_img_cutout' % ref_band].data.shape[1] * pixel_regrade_factor))

        # Now we have to get the coordinates in the corners to estimate the width and hence the pixel scale!
        world_coords_lower_left = cutout_dict['%s_img_cutout' % ref_band].wcs.pixel_to_world(0, 0)
        world_coords_upper_right = cutout_dict['%s_img_cutout' % ref_band].wcs.pixel_to_world(ref_shape_old[1],
                                                                                              ref_shape_old[0])

        ra_min = world_coords_lower_left.ra.degree
        ra_max = world_coords_upper_right.ra.degree
        dec_min = world_coords_lower_left.dec.degree
        dec_max = world_coords_upper_right.dec.degree

        new_wcs = helper_func.CoordTools.construct_wcs(
            ra_min=ra_min, ra_max=ra_max, dec_min=dec_max, dec_max=dec_min,
            img_shape=new_shape, quadratic_image=False,
            ctype=[cutout_dict['%s_img_cutout' % ref_band].wcs.to_header()['CTYPE1'],
                   cutout_dict['%s_img_cutout' % ref_band].wcs.to_header()['CTYPE2']])

        # reproject all three colorsto the new WCS
        img_data_red = helper_func.CoordTools.reproject_image(data=cutout_dict['%s_img_cutout' % red_band].data,
                                                              wcs=cutout_dict['%s_img_cutout' % red_band].wcs,
                                                              new_wcs=new_wcs, new_shape=new_shape)
        img_data_green = helper_func.CoordTools.reproject_image(data=cutout_dict['%s_img_cutout' % green_band].data,
                                                                wcs=cutout_dict['%s_img_cutout' % green_band].wcs,
                                                                new_wcs=new_wcs, new_shape=new_shape)
        img_data_blue = helper_func.CoordTools.reproject_image(data=cutout_dict['%s_img_cutout' % blue_band].data,
                                                               wcs=cutout_dict['%s_img_cutout' % blue_band].wcs,
                                                               new_wcs=new_wcs, new_shape=new_shape)

        img_data_red[img_data_red == 0] = np.nan
        img_data_green[img_data_green == 0] = np.nan
        img_data_blue[img_data_blue == 0] = np.nan

        hst_rgb = plotting_tools.ImgTools.get_rgb_img(data_r=img_data_red, data_g=img_data_green, data_b=img_data_blue,
                                                      **rgb_img_kwargs)

        return hst_rgb, new_wcs

    def plot_zoom_in_panel_group(self, fig, fig_dict, ra, dec, obs_list=None, nrows=2, ncols=3):

        if obs_list is None:
            obs_list = ['hst_broad_band', 'hst_ha', 'astrosat', 'nircam', 'miri', 'alma']

        row_idx, col_idx = nrows - 1, 0
        for idx, obs_type in enumerate(obs_list):

            if obs_type in ['hst_broad_band', 'hst_ha', 'nircam', 'miri']:

                obs = obs_type.split('_')[0]
                if obs == 'hst':
                    # check if object has hst Ha observation
                    if (fig_dict['%s_red_band' % obs_type] == 'Ha') & (
                    not ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_target_name)):
                        continue
                    band_list = [ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                           filter_name=fig_dict[
                                                                               '%s_red_band' % obs_type]),
                                 ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                           filter_name=fig_dict[
                                                                               '%s_green_band' % obs_type]),
                                 ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                                           filter_name=fig_dict[
                                                                               '%s_blue_band' % obs_type])]
                else:
                    band_list = [fig_dict['%s_red_band' % obs_type], fig_dict['%s_green_band' % obs_type],
                                 fig_dict['%s_blue_band' % obs_type]]

                # check if object is covered
                if (self.check_coords_covered_by_band(
                        telescope=obs, ra=ra, dec=dec, band=band_list[0], max_dist_dist2hull_arcsec=2) &
                        self.check_coords_covered_by_band(
                            telescope=obs, ra=ra, dec=dec, band=band_list[1], max_dist_dist2hull_arcsec=2) &
                        self.check_coords_covered_by_band(
                            telescope=obs, ra=ra, dec=dec, band=band_list[2], max_dist_dist2hull_arcsec=2)):
                    self.load_obs_bands(band_list=band_list, flux_unit='MJy/sr', load_err=False)

                    img_zoom_in, wcs_zoom_in = self.get_rgb_zoom_in(ra=ra, dec=dec,
                                                                    cutout_size=fig_dict['env_cutout_size'],
                                                                    band_red=band_list[0],
                                                                    band_green=band_list[1], band_blue=band_list[2])

                    ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['zoom_in_left_align'],
                        bottom_align=fig_dict['zoom_in_bottom_align'],
                        width=fig_dict['zoom_in_width'],
                        height=fig_dict['zoom_in_height'],
                        space_vertical=fig_dict['zoom_in_space_vertical'],
                        space_horizontal=fig_dict['zoom_in_space_horizontal'],
                        row_idx=row_idx, col_idx=col_idx, projection=wcs_zoom_in)

                    ax_zoom_in.imshow(img_zoom_in)
                    # arrange axis
                    # add text
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text=obs.upper(),
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='white', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_red_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='red', x_frac=0.02, y_frac=0.91,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_green_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='green', x_frac=0.02, y_frac=0.84,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_blue_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='blue', x_frac=0.02, y_frac=0.77,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')

                    plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                    dec_tick_label=False,
                                                                    ra_axis_label=' ',
                                                                    dec_axis_label=' ',
                                                                    ra_minpad=0.8, dec_minpad=0.8,
                                                                    ra_tick_color='white', dec_tick_color='white',
                                                                    ra_label_color='k', dec_label_color='k',
                                                                    fontsize=fig_dict['zoom_in_label_size'],
                                                                    labelsize=fig_dict['zoom_in_label_size'],
                                                                    ra_minor_ticks=True, dec_minor_ticks=True)

                    if (col_idx == 0) & (row_idx == nrows - 1):
                        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                            ax=ax_zoom_in, img_shape=img_zoom_in.shape, wcs=wcs_zoom_in,
                            bar_length=fig_dict['zoom_in_scale_bar_length_1'], length_unit='pc',
                            phangs_target=self.phot_target_name,
                            bar_color='tab:red', text_color='tab:red',
                            line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

                        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                            ax=ax_zoom_in, img_shape=img_zoom_in.shape, wcs=wcs_zoom_in,
                            bar_length=fig_dict['zoom_in_scale_bar_length_2'], length_unit='arcsec',
                            phangs_target=self.phot_target_name,
                            bar_color='tab:red', text_color='tab:red',
                            line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                            va='top', ha='right', x_offset=0.05, y_offset=0.15, text_y_offset_diff=0.01)

            elif obs_type == 'astrosat':
                # check if object is covered
                if ObsTools.check_astrosat_obs(target=self.phot_astrosat_target_name):
                    astrosat_band_list = ObsTools.get_astrosat_obs_band_list(
                        target=self.phot_astrosat_target_name)
                    if fig_dict['astrosat_band'] in astrosat_band_list:
                        astrosat_band = fig_dict['astrosat_band']
                    else:
                        astrosat_band = astrosat_band_list[0]
                else:
                    continue

                if self.check_coords_covered_by_band(telescope="astrosat", ra=ra, dec=dec, band=astrosat_band,
                                                     max_dist_dist2hull_arcsec=2):
                    self.load_obs_bands(band_list=astrosat_band, flux_unit='erg A-1 cm-2 s-1', load_err=False)
                    cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec,
                                                            cutout_size=fig_dict['env_cutout_size'],
                                                            band_list=[astrosat_band])
                    ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['zoom_in_left_align'],
                        bottom_align=fig_dict['zoom_in_bottom_align'],
                        width=fig_dict['zoom_in_width'],
                        height=fig_dict['zoom_in_height'],
                        space_vertical=fig_dict['zoom_in_space_vertical'],
                        space_horizontal=fig_dict['zoom_in_space_horizontal'],
                        row_idx=row_idx, col_idx=col_idx,
                        projection=cutout_dict['%s_img_cutout' % astrosat_band].wcs)

                    min_astrosat_value = np.nanmin(cutout_dict['%s_img_cutout' % astrosat_band].data)
                    max_astrosat_value = np.nanmax(cutout_dict['%s_img_cutout' % astrosat_band].data)
                    if min_astrosat_value <= 0:
                        min_astrosat_value = max_astrosat_value / 100
                    if fig_dict['astrosat_norm'] == 'log':
                        norm = LogNorm(min_astrosat_value, max_astrosat_value)
                    else:
                        norm = Normalize(min_astrosat_value, max_astrosat_value)

                    ax_zoom_in.imshow(cutout_dict['%s_img_cutout' % astrosat_band].data, norm=norm,
                                      cmap=fig_dict['astrosat_cmap'])

                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text='ASTROSAT',
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='white', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='red')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=astrosat_band,
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='red', x_frac=0.02, y_frac=0.91,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='red')
                    plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                    dec_tick_label=False,
                                                                    ra_axis_label=' ',
                                                                    dec_axis_label=' ',
                                                                    ra_minpad=0.8, dec_minpad=0.8,
                                                                    ra_tick_color='white', dec_tick_color='white',
                                                                    ra_label_color='k', dec_label_color='k',
                                                                    fontsize=fig_dict['zoom_in_label_size'],
                                                                    labelsize=fig_dict['zoom_in_label_size'],
                                                                    ra_minor_ticks=True, dec_minor_ticks=True)
                    ax_cbar_astrosat = fig.add_axes([fig_dict['astrosat_cbar_left_align'],
                                                     fig_dict['astrosat_cbar_bottom_align'],
                                                     fig_dict['astrosat_cbar_width'],
                                                     fig_dict['astrosat_cbar_height']])
                    plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_astrosat, cmap=fig_dict['astrosat_cmap'],
                                                             norm=norm,
                                                             cbar_label=r'$\phi$ [erg $\AA^{-1}$ cm$^{-2}$ s$^{-2}$}',
                                                             fontsize=fig_dict['zoom_in_label_size'],
                                                             ticks=None, labelpad=2, tick_width=2,
                                                             orientation='vertical',
                                                             extend='neither')

            elif obs_type == 'alma':

                # check if target is observed
                if self.gas_target_name in obs_info.full_alma_galaxy_list:

                    # check if object is covered
                    if self.check_coords_covered_by_alma(ra=ra, dec=dec, res=fig_dict['alma_res'],
                                                         max_dist_dist2hull_arcsec=2):

                        alma_h2_map_data, alma_h2_map_wcs = self.get_alma_h2_map(
                            res=fig_dict['alma_res'], alpha_co_method=fig_dict['alma_alpha_co_method'])
                        # get cutout
                        alma_h2_map_cutout = helper_func.CoordTools.get_img_cutout(
                            img=alma_h2_map_data, wcs=alma_h2_map_wcs, coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                            cutout_size=fig_dict['env_cutout_size'])
                        # make sure the alma data is covered:
                        if (np.all(alma_h2_map_cutout.data == 0) | np.all(np.isnan(alma_h2_map_cutout.data)) |
                                np.all(np.isnan(alma_h2_map_cutout.data) + (alma_h2_map_cutout.data == 0))):
                            print('no alma coverage')
                            break
                        ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['zoom_in_left_align'],
                            bottom_align=fig_dict['zoom_in_bottom_align'],
                            width=fig_dict['zoom_in_width'],
                            height=fig_dict['zoom_in_height'],
                            space_vertical=fig_dict['zoom_in_space_vertical'],
                            space_horizontal=fig_dict['zoom_in_space_horizontal'],
                            row_idx=row_idx, col_idx=col_idx,
                            projection=alma_h2_map_cutout.wcs)

                        min_alma_value = np.nanmin(alma_h2_map_cutout.data)
                        max_alma_value = np.nanmax(alma_h2_map_cutout.data)

                        if min_alma_value <= 0:
                            min_alma_value = max_alma_value / 100
                        if fig_dict['alma_norm'] == 'log':
                            norm = LogNorm(min_alma_value, max_alma_value)
                        else:
                            norm = Normalize(min_alma_value, max_alma_value)

                        ax_zoom_in.imshow(alma_h2_map_cutout.data, norm=norm, cmap=fig_dict['alma_cmap'])
                        plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text='ALMA',
                                                                       fontsize=fig_dict['zoom_in_title_font_size'],
                                                                       text_color='white', x_frac=0.02, y_frac=0.98,
                                                                       horizontal_alignment='left',
                                                                       vertical_alignment='top',
                                                                       path_eff=True, path_err_linewidth=3,
                                                                       path_eff_color='red')
                        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                        dec_tick_label=False,
                                                                        ra_axis_label=' ',
                                                                        dec_axis_label=' ',
                                                                        ra_minpad=0.8, dec_minpad=0.8,
                                                                        ra_tick_color='white', dec_tick_color='white',
                                                                        ra_label_color='k', dec_label_color='k',
                                                                        fontsize=fig_dict['zoom_in_label_size'],
                                                                        labelsize=fig_dict['zoom_in_label_size'],
                                                                        ra_minor_ticks=True, dec_minor_ticks=True)
                        ax_cbar_alma = fig.add_axes([fig_dict['alma_cbar_left_align'],
                                                     fig_dict['alma_cbar_bottom_align'],
                                                     fig_dict['alma_cbar_width'],
                                                     fig_dict['alma_cbar_height']])
                        plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_alma, cmap=fig_dict['alma_cmap'],
                                                                 norm=norm,
                                                                 cbar_label=r'log($\Sigma_{\rm H2}$/[M$_{\odot}$ kpc$^{-2}$])',
                                                                 fontsize=fig_dict['zoom_in_label_size'],
                                                                 ticks=None, labelpad=2, tick_width=2,
                                                                 orientation='vertical',
                                                                 extend='neither')

            col_idx += 1
            if col_idx == ncols:
                col_idx = 0
                row_idx -= 1

    def plot_zoom_in_panel_group_extra_nircam(self, fig, fig_dict, ra, dec, obs_list=None, nrows=2, ncols=3):

        if obs_list is None:
            obs_list = ['hst_broad_band', 'hst_ha', 'nircam1', 'nircam2', 'nircam3', 'nircam4', 'nircam5', 'miri1',
                        'miri2',]


        row_idx, col_idx = nrows - 1, 0
        for idx, obs_type in enumerate(obs_list):

            if obs_type in ['hst_broad_band', 'hst_ha', 'nircam1', 'nircam2', 'nircam3', 'nircam4', 'nircam5',
                            'miri1', 'miri2']:

                if obs_type in ['nircam1', 'nircam2', 'nircam3', 'nircam4', 'nircam5']:
                    obs = 'jwst'
                    instrument = 'nircam'
                elif obs_type in ['miri1', 'miri2']:
                    obs = 'jwst'
                    instrument = 'miri'
                elif obs_type.split('_')[0] == 'hst':
                    obs = 'hst'
                    instrument = None
                else:
                    raise KeyError('Obs type not understand')

                if obs == 'hst':
                    # check if object has hst Ha observation
                    if (fig_dict['%s_red_band' % obs_type] == 'Ha') & (
                    not ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_target_name)):
                        continue
                    band_list = [ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                               filter_name=fig_dict['%s_red_band' % obs_type]),
                                 ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                               filter_name=fig_dict['%s_green_band' % obs_type]),
                                 ObsTools.filter_name2hst_band(target=self.phot_hst_target_name,
                                                               filter_name=fig_dict['%s_blue_band' % obs_type])]
                else:
                    band_list = [fig_dict['%s_red_band' % obs_type], fig_dict['%s_green_band' % obs_type],
                                 fig_dict['%s_blue_band' % obs_type]]

                # check if object is covered
                if (self.check_coords_covered_by_band(
                        obs=obs, ra=ra, dec=dec, band=band_list[0], instrument=instrument,  max_dist_dist2hull_arcsec=0.5) &
                        self.check_coords_covered_by_band(
                            obs=obs, ra=ra, dec=dec, band=band_list[1], instrument=instrument,  max_dist_dist2hull_arcsec=0.5) &
                        self.check_coords_covered_by_band(
                            obs=obs, ra=ra, dec=dec, band=band_list[2], instrument=instrument,  max_dist_dist2hull_arcsec=0.5)):
                    self.load_obs_bands(band_list=band_list, flux_unit='MJy/sr', load_err=False)

                    img_zoom_in, wcs_zoom_in = self.get_rgb_zoom_in(ra=ra, dec=dec,
                                                                    cutout_size=fig_dict['env_cutout_size'],
                                                                    band_red=band_list[0],
                                                                    band_green=band_list[1], band_blue=band_list[2])

                    ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['zoom_in_left_align'],
                        bottom_align=fig_dict['zoom_in_bottom_align'],
                        width=fig_dict['zoom_in_width'],
                        height=fig_dict['zoom_in_height'],
                        space_vertical=fig_dict['zoom_in_space_vertical'],
                        space_horizontal=fig_dict['zoom_in_space_horizontal'],
                        row_idx=row_idx, col_idx=col_idx, projection=wcs_zoom_in)

                    ax_zoom_in.imshow(img_zoom_in)
                    # arrange axis
                    # add text
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text=obs.upper(),
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='white', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_red_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='red', x_frac=0.02, y_frac=0.91,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_green_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='green', x_frac=0.02, y_frac=0.84,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=fig_dict['%s_blue_band' % obs_type],
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='blue', x_frac=0.02, y_frac=0.77,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='white')

                    plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                    dec_tick_label=False,
                                                                    ra_axis_label=' ',
                                                                    dec_axis_label=' ',
                                                                    ra_minpad=0.8, dec_minpad=0.8,
                                                                    ra_tick_color='white', dec_tick_color='white',
                                                                    ra_label_color='k', dec_label_color='k',
                                                                    fontsize=fig_dict['zoom_in_label_size'],
                                                                    labelsize=fig_dict['zoom_in_label_size'],
                                                                    ra_minor_ticks=True, dec_minor_ticks=True)

                    if (col_idx == 0) & (row_idx == nrows - 1):
                        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                            ax=ax_zoom_in, img_shape=img_zoom_in.shape, wcs=wcs_zoom_in,
                            bar_length=fig_dict['zoom_in_scale_bar_length_1'], length_unit='pc',
                            phangs_target=self.phot_target_name,
                            bar_color='tab:red', text_color='tab:red',
                            line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

                        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                            ax=ax_zoom_in, img_shape=img_zoom_in.shape, wcs=wcs_zoom_in,
                            bar_length=fig_dict['zoom_in_scale_bar_length_2'], length_unit='arcsec',
                            phangs_target=self.phot_target_name,
                            bar_color='tab:red', text_color='tab:red',
                            line_width=4, fontsize=fig_dict['zoom_in_label_size'],
                            va='top', ha='right', x_offset=0.05, y_offset=0.15, text_y_offset_diff=0.01)

            elif obs_type == 'astrosat':
                # check if object is covered
                if ObsTools.check_astrosat_obs(target=self.phot_astrosat_target_name):
                    astrosat_band_list = ObsTools.get_astrosat_obs_band_list(
                        target=self.phot_astrosat_target_name)
                    if fig_dict['astrosat_band'] in astrosat_band_list:
                        astrosat_band = fig_dict['astrosat_band']
                    else:
                        astrosat_band = astrosat_band_list[0]
                else:
                    continue

                if self.check_coords_covered_by_band(telescope="astrosat", ra=ra, dec=dec, band=astrosat_band,
                                                     max_dist_dist2hull_arcsec=2):
                    self.load_obs_bands(band_list=astrosat_band, flux_unit='erg A-1 cm-2 s-1', load_err=False)
                    cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec,
                                                            cutout_size=fig_dict['env_cutout_size'],
                                                            band_list=[astrosat_band])
                    ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['zoom_in_left_align'],
                        bottom_align=fig_dict['zoom_in_bottom_align'],
                        width=fig_dict['zoom_in_width'],
                        height=fig_dict['zoom_in_height'],
                        space_vertical=fig_dict['zoom_in_space_vertical'],
                        space_horizontal=fig_dict['zoom_in_space_horizontal'],
                        row_idx=row_idx, col_idx=col_idx,
                        projection=cutout_dict['%s_img_cutout' % astrosat_band].wcs)

                    min_astrosat_value = np.nanmin(cutout_dict['%s_img_cutout' % astrosat_band].data)
                    max_astrosat_value = np.nanmax(cutout_dict['%s_img_cutout' % astrosat_band].data)
                    if min_astrosat_value <= 0:
                        min_astrosat_value = max_astrosat_value / 100
                    if fig_dict['astrosat_norm'] == 'log':
                        norm = LogNorm(min_astrosat_value, max_astrosat_value)
                    else:
                        norm = Normalize(min_astrosat_value, max_astrosat_value)

                    ax_zoom_in.imshow(cutout_dict['%s_img_cutout' % astrosat_band].data, norm=norm,
                                      cmap=fig_dict['astrosat_cmap'])

                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text='ASTROSAT',
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='white', x_frac=0.02, y_frac=0.98,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='red')
                    plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in,
                                                                   text=astrosat_band,
                                                                   fontsize=fig_dict['zoom_in_title_font_size'],
                                                                   text_color='red', x_frac=0.02, y_frac=0.91,
                                                                   horizontal_alignment='left',
                                                                   vertical_alignment='top',
                                                                   path_eff=True, path_err_linewidth=3,
                                                                   path_eff_color='red')
                    plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                    dec_tick_label=False,
                                                                    ra_axis_label=' ',
                                                                    dec_axis_label=' ',
                                                                    ra_minpad=0.8, dec_minpad=0.8,
                                                                    ra_tick_color='white', dec_tick_color='white',
                                                                    ra_label_color='k', dec_label_color='k',
                                                                    fontsize=fig_dict['zoom_in_label_size'],
                                                                    labelsize=fig_dict['zoom_in_label_size'],
                                                                    ra_minor_ticks=True, dec_minor_ticks=True)
                    ax_cbar_astrosat = fig.add_axes([fig_dict['astrosat_cbar_left_align'],
                                                     fig_dict['astrosat_cbar_bottom_align'],
                                                     fig_dict['astrosat_cbar_width'],
                                                     fig_dict['astrosat_cbar_height']])
                    plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_astrosat, cmap=fig_dict['astrosat_cmap'],
                                                             norm=norm,
                                                             cbar_label=r'$\phi$ [erg $\AA^{-1}$ cm$^{-2}$ s$^{-2}$}',
                                                             fontsize=fig_dict['zoom_in_label_size'],
                                                             ticks=None, labelpad=2, tick_width=2,
                                                             orientation='vertical',
                                                             extend='neither')

            elif obs_type == 'alma':

                # check if target is observed
                if self.gas_target_name in obs_info.full_alma_galaxy_list:

                    # check if object is covered
                    if self.check_coords_covered_by_alma(ra=ra, dec=dec, res=fig_dict['alma_res'],
                                                         max_dist_dist2hull_arcsec=2):

                        alma_h2_map_data, alma_h2_map_wcs = self.get_alma_h2_map(
                            res=fig_dict['alma_res'], alpha_co_method=fig_dict['alma_alpha_co_method'])
                        # get cutout
                        alma_h2_map_cutout = helper_func.CoordTools.get_img_cutout(
                            img=alma_h2_map_data, wcs=alma_h2_map_wcs, coord=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                            cutout_size=fig_dict['env_cutout_size'])
                        # make sure the alma data is covered:
                        if (np.all(alma_h2_map_cutout.data == 0) | np.all(np.isnan(alma_h2_map_cutout.data)) |
                                np.all(np.isnan(alma_h2_map_cutout.data) + (alma_h2_map_cutout.data == 0))):
                            print('no alma coverage')
                            break
                        ax_zoom_in = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['zoom_in_left_align'],
                            bottom_align=fig_dict['zoom_in_bottom_align'],
                            width=fig_dict['zoom_in_width'],
                            height=fig_dict['zoom_in_height'],
                            space_vertical=fig_dict['zoom_in_space_vertical'],
                            space_horizontal=fig_dict['zoom_in_space_horizontal'],
                            row_idx=row_idx, col_idx=col_idx,
                            projection=alma_h2_map_cutout.wcs)

                        min_alma_value = np.nanmin(alma_h2_map_cutout.data)
                        max_alma_value = np.nanmax(alma_h2_map_cutout.data)

                        if min_alma_value <= 0:
                            min_alma_value = max_alma_value / 100
                        if fig_dict['alma_norm'] == 'log':
                            norm = LogNorm(min_alma_value, max_alma_value)
                        else:
                            norm = Normalize(min_alma_value, max_alma_value)

                        ax_zoom_in.imshow(alma_h2_map_cutout.data, norm=norm, cmap=fig_dict['alma_cmap'])
                        plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in, text='ALMA',
                                                                       fontsize=fig_dict['zoom_in_title_font_size'],
                                                                       text_color='white', x_frac=0.02, y_frac=0.98,
                                                                       horizontal_alignment='left',
                                                                       vertical_alignment='top',
                                                                       path_eff=True, path_err_linewidth=3,
                                                                       path_eff_color='red')
                        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_zoom_in, ra_tick_label=False,
                                                                        dec_tick_label=False,
                                                                        ra_axis_label=' ',
                                                                        dec_axis_label=' ',
                                                                        ra_minpad=0.8, dec_minpad=0.8,
                                                                        ra_tick_color='white', dec_tick_color='white',
                                                                        ra_label_color='k', dec_label_color='k',
                                                                        fontsize=fig_dict['zoom_in_label_size'],
                                                                        labelsize=fig_dict['zoom_in_label_size'],
                                                                        ra_minor_ticks=True, dec_minor_ticks=True)
                        ax_cbar_alma = fig.add_axes([fig_dict['alma_cbar_left_align'],
                                                     fig_dict['alma_cbar_bottom_align'],
                                                     fig_dict['alma_cbar_width'],
                                                     fig_dict['alma_cbar_height']])
                        plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_alma, cmap=fig_dict['alma_cmap'],
                                                                 norm=norm,
                                                                 cbar_label=r'log($\Sigma_{\rm H2}$/[M$_{\odot}$ kpc$^{-2}$])',
                                                                 fontsize=fig_dict['zoom_in_label_size'],
                                                                 ticks=None, labelpad=2, tick_width=2,
                                                                 orientation='vertical',
                                                                 extend='neither')

            col_idx += 1
            if col_idx == ncols:
                col_idx = 0
                row_idx -= 1

    def plot_x_ray_zoom_in(self, fig, fig_dict, ra, dec):

        # load X-ray data
        self.load_chandra_data(energy='0p5to2')
        self.load_chandra_data(energy='2to7')

        # get cutout
        cutout_0p5to2 = helper_func.CoordTools.get_img_cutout(
            img=self.x_data['0p5to2_data_img'], wcs=self.x_data['0p5to2_wcs_img'],
            coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), cutout_size=fig_dict['env_cutout_size'])

        cutout_2to7 = helper_func.CoordTools.get_img_cutout(
            img=self.x_data['2to7_data_img'], wcs=self.x_data['2to7_wcs_img'],
            coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), cutout_size=fig_dict['env_cutout_size'])


        ax_zoom_in_0p5to2 = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['x_ray_zoom_in_left_align'],
                bottom_align=fig_dict['x_ray_zoom_in_bottom_align'],
                width=fig_dict['x_ray_zoom_in_width'],
                height=fig_dict['x_ray_zoom_in_height'],
                space_vertical=fig_dict['x_ray_zoom_in_space_vertical'],
                space_horizontal=fig_dict['x_ray_zoom_in_space_horizontal'],
                row_idx=0, col_idx=0,
                projection=cutout_0p5to2.wcs)

        ax_cbar_0p5to2 = fig.add_axes([fig_dict['0p5to2_cbar_left_align'],
                                       fig_dict['0p5to2_cbar_bottom_align'],
                                       fig_dict['0p5to2_cbar_width'],
                                       fig_dict['0p5to2_cbar_height']])

        ax_zoom_in_2to7 = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['x_ray_zoom_in_left_align'],
                bottom_align=fig_dict['x_ray_zoom_in_bottom_align'],
                width=fig_dict['x_ray_zoom_in_width'],
                height=fig_dict['x_ray_zoom_in_height'],
                space_vertical=fig_dict['x_ray_zoom_in_space_vertical'],
                space_horizontal=fig_dict['x_ray_zoom_in_space_horizontal'],
                row_idx=0, col_idx=1,
                projection=cutout_2to7.wcs)

        ax_cbar_2to7 = fig.add_axes([fig_dict['2to7_cbar_left_align'],
                                       fig_dict['2to7_cbar_bottom_align'],
                                       fig_dict['2to7_cbar_width'],
                                       fig_dict['2to7_cbar_height']])

        min_0p5to2 = np.min(cutout_0p5to2.data)
        min_2to7 = np.min(cutout_2to7.data)

        max_0p5to2 = np.max(cutout_0p5to2.data)
        max_2to7 = np.max(cutout_2to7.data)

        if max_0p5to2 != 0:
            norm_0p5to2 = plotting_tools.ColorBarTools.compute_cbar_norm(vmin_vmax=(min_0p5to2 + 0.1, max_0p5to2), log_scale=True)
            ax_zoom_in_0p5to2.imshow(cutout_0p5to2.data, cmap=fig_dict['0p5to2_cmap'], norm=norm_0p5to2)

            plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_0p5to2, cmap=fig_dict['0p5to2_cmap'],
                                                     norm=norm_0p5to2, cbar_label=r'Counts', fontsize=fig_dict['zoom_in_label_size'], ticks=None, labelpad=2, tick_width=2,
                            orientation='horizontal', top_lable=True, label_color='k',
                            extend='neither')
        else:
            ax_zoom_in_0p5to2.imshow(cutout_0p5to2.data, cmap=fig_dict['0p5to2_cmap'])

        if max_2to7 != 0:
            norm_2to7 = plotting_tools.ColorBarTools.compute_cbar_norm(vmin_vmax=(min_2to7 + 0.1, max_2to7), log_scale=True)
            ax_zoom_in_2to7.imshow(cutout_2to7.data, cmap=fig_dict['2to7_cmap'], norm=norm_2to7)

            plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_2to7, cmap=fig_dict['2to7_cmap'],
                                                     norm=norm_2to7, cbar_label=r'Counts', fontsize=fig_dict['zoom_in_label_size'], ticks=None, labelpad=2, tick_width=2,
                            orientation='horizontal', top_lable=True, label_color='k',
                            extend='neither')
        else:
            ax_zoom_in_2to7.imshow(cutout_2to7.data, cmap=fig_dict['2to7_cmap'])


        # norm_2to7 = plotting_tools.ColorBarTools.compute_cbar_norm(vmin_vmax=(min_2to7 + 0.1, max_2to7), log_scale=True)
        #
        # ax_zoom_in_2to7.imshow(cutout_2to7.data, cmap=fig_dict['2to7_cmap'], norm=norm_2to7)
        #
        #
        #
        #
        # plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_2to7, cmap=fig_dict['2to7_cmap'],
        #                                          norm=norm_2to7, cbar_label=r'Counts', fontsize=fig_dict['zoom_in_label_size'], ticks=None, labelpad=2, tick_width=2,
        #                 orientation='horizontal', top_lable=True, label_color='k',
        #                 extend='neither')

        plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in_0p5to2, text='0.5 - 2 eV',
                                                           fontsize=fig_dict['zoom_in_title_font_size'],
                                                           text_color='black', x_frac=0.02, y_frac=0.98,
                                                           horizontal_alignment='left',
                                                           vertical_alignment='top',
                                                           path_eff=True, path_err_linewidth=3,
                                                           path_eff_color='red')

        plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in_2to7, text='2 - 7 eV',
                                                           fontsize=fig_dict['zoom_in_title_font_size'],
                                                           text_color='black', x_frac=0.02, y_frac=0.98,
                                                           horizontal_alignment='left',
                                                           vertical_alignment='top',
                                                           path_eff=True, path_err_linewidth=3,
                                                           path_eff_color='red')

        plotting_tools.WCSPlottingTools.arr_axis_params(
            ax=ax_zoom_in_0p5to2, ra_tick_label=False, dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ')
        plotting_tools.WCSPlottingTools.arr_axis_params(
            ax=ax_zoom_in_2to7, ra_tick_label=False, dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ')

    def plot_radio_cont_zoom_in(self, fig, fig_dict, ra, dec, radio_ellipse_dict=None):

        # load X-ray data
        self.load_radio_data(band='L', map='tt0')
        self.load_radio_data(band='L', map='alpha')
        # get cutout

        cutout_tt0 = helper_func.CoordTools.get_img_cutout(
            img=self.radio_data['L_tt0_data_img'], wcs=self.radio_data['L_tt0_wcs_img'],
            coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), cutout_size=fig_dict['env_cutout_size'])

        cutout_alpha = helper_func.CoordTools.get_img_cutout(
            img=self.radio_data['L_alpha_data_img'], wcs=self.radio_data['L_alpha_wcs_img'],
            coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), cutout_size=fig_dict['env_cutout_size'])


        ax_zoom_in_tt0 = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['radio_zoom_in_left_align'],
                bottom_align=fig_dict['radio_zoom_in_bottom_align'],
                width=fig_dict['radio_zoom_in_width'],
                height=fig_dict['radio_zoom_in_height'],
                space_vertical=fig_dict['radio_zoom_in_space_vertical'],
                space_horizontal=fig_dict['radio_zoom_in_space_horizontal'],
                row_idx=0, col_idx=2,
                projection=cutout_tt0.wcs)

        ax_zoom_in_alpha = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['radio_zoom_in_left_align'],
                bottom_align=fig_dict['radio_zoom_in_bottom_align'],
                width=fig_dict['radio_zoom_in_width'],
                height=fig_dict['radio_zoom_in_height'],
                space_vertical=fig_dict['radio_zoom_in_space_vertical'],
                space_horizontal=fig_dict['radio_zoom_in_space_horizontal'],
                row_idx=0, col_idx=3,
                projection=cutout_alpha.wcs)

        ax_cbar_alpha = fig.add_axes([fig_dict['alpha_cbar_left_align'],
                                       fig_dict['alpha_cbar_bottom_align'],
                                       fig_dict['alpha_cbar_width'],
                                       fig_dict['alpha_cbar_height']])


        norm_tt0 = plotting_tools.ColorBarTools.compute_cbar_norm(vmin_vmax=(np.nanmin(cutout_tt0.data),
                                                                             np.nanmax(cutout_tt0.data)), log_scale=True)
        norm_alpha = plotting_tools.ColorBarTools.compute_cbar_norm(vmin_vmax=(-2, 0.5), log_scale=False)


        ax_zoom_in_tt0.imshow(cutout_tt0.data, cmap=fig_dict['tt0_cmap'], norm=norm_tt0)
        ax_zoom_in_alpha.imshow(cutout_alpha.data, cmap=fig_dict['alpha_cmap'], norm=norm_alpha)


        plotting_tools.ColorBarTools.create_cbar(ax_cbar=ax_cbar_alpha, cmap=fig_dict['alpha_cmap'],
                                                 norm=norm_alpha, cbar_label=r'$\alpha$',
                                                 fontsize=fig_dict['zoom_in_label_size'], ticks=None, labelpad=2, tick_width=2,
                        orientation='vertical', top_lable=True, label_color='k',
                        extend='neither')


        plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in_tt0, text='Radio continuum',
                                                           fontsize=fig_dict['zoom_in_title_font_size'],
                                                           text_color='red', x_frac=0.02, y_frac=0.98,
                                                           horizontal_alignment='left',
                                                           vertical_alignment='top',
                                                           path_eff=True, path_err_linewidth=3,
                                                           path_eff_color='red')

        plotting_tools.StrTools.display_text_in_corner(ax=ax_zoom_in_alpha, text='Spectral index',
                                                           fontsize=fig_dict['zoom_in_title_font_size'],
                                                           text_color='black', x_frac=0.02, y_frac=0.98,
                                                           horizontal_alignment='left',
                                                           vertical_alignment='top',
                                                           path_eff=True, path_err_linewidth=3,
                                                           path_eff_color='red')

        plotting_tools.WCSPlottingTools.arr_axis_params(
            ax=ax_zoom_in_tt0, ra_tick_label=False, dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ')
        plotting_tools.WCSPlottingTools.arr_axis_params(
            ax=ax_zoom_in_alpha, ra_tick_label=False, dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ')

        if radio_ellipse_dict is not None:

            ellipse_sky = EllipseSkyRegion(center=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), width=radio_ellipse_dict['min_axis']*u.deg,

                                   height=radio_ellipse_dict['maj_axis']*u.deg, angle=radio_ellipse_dict['angle']*u.deg)

            ellipse_pixel_zoom_in = ellipse_sky.to_pixel(cutout_tt0.wcs)

            patch = ellipse_pixel_zoom_in.as_artist(facecolor=radio_ellipse_dict['facecolor'],
                                                    edgecolor=radio_ellipse_dict['edgecolor'], linewidth=radio_ellipse_dict['linewidth'], linestyle='-')

            ax_zoom_in_tt0.add_patch(patch)




    def get_rgb_zoom_in(self, ra, dec, cutout_size, band_red, band_green, band_blue, ref_filter='blue', **kwargs):
        """
        Function to create an RGB image of a zoom in region of PHANGS observations

        Parameters
        ----------
        ra, dec : float
            coordinates in degree
        cutout_size: tuple
            cutout size in arcsec
        circle_rad : float
            radius of circle in which the object of interest is located
        band_red, band_green, band_blue : str
            filter names
        ref_filter : str
            can be blue, green or red. In case the images are not the same size they get reprojected to one frame

        Returns
        -------
        rgb_image, wcs : ``numpy.ndarray``, ``astropy.wcs.WCS``
        """

        self.load_obs_bands(band_list=[band_red, band_green, band_blue],
                               flux_unit='MJy/sr', load_err=False)

        cutout = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size,
                                           band_list=[band_red, band_green, band_blue])

        if ((cutout['%s_img_cutout' % band_red].data is None) or
                (cutout['%s_img_cutout' % band_green].data is None) or
                (cutout['%s_img_cutout' % band_blue].data is None)):
            return None, None

        ref_wcs = cutout['%s_img_cutout' % eval('band_%s' % ref_filter)].wcs

        if not (cutout['%s_img_cutout' % band_red].data.shape ==
                cutout['%s_img_cutout' % band_green].data.shape ==
                cutout['%s_img_cutout' % band_blue].data.shape):
            new_shape = cutout['%s_img_cutout' % eval('band_%s' % ref_filter)].data.shape
            if ref_filter == 'red':
                cutout_data_red = cutout['%s_img_cutout' % band_red].data
            else:
                cutout_data_red = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_red].data,
                                                                         wcs=cutout['%s_img_cutout' % band_red].wcs,
                                                                         new_wcs=ref_wcs, new_shape=new_shape)
            if ref_filter == 'green':
                cutout_data_green = cutout['%s_img_cutout' % band_green].data
            else:
                cutout_data_green = helper_func.CoordTools.reproject_image(
                    data=cutout['%s_img_cutout' % band_green].data,
                    wcs=cutout['%s_img_cutout' % band_green].wcs,
                    new_wcs=ref_wcs, new_shape=new_shape)
            if ref_filter == 'blue':
                cutout_data_blue = cutout['%s_img_cutout' % band_blue].data
            else:
                cutout_data_blue = helper_func.CoordTools.reproject_image(data=cutout['%s_img_cutout' % band_blue].data,
                                                                          wcs=cutout['%s_img_cutout' % band_blue].wcs,
                                                                          new_wcs=ref_wcs, new_shape=new_shape)
        else:
            cutout_data_red = cutout['%s_img_cutout' % band_red].data
            cutout_data_green = cutout['%s_img_cutout' % band_green].data
            cutout_data_blue = cutout['%s_img_cutout' % band_blue].data

        cutout_rgb_img = plotting_tools.ImgTools.get_rgb_img(data_r=cutout_data_red,
                                                             data_g=cutout_data_green,
                                                             data_b=cutout_data_blue, **kwargs)

        return cutout_rgb_img, ref_wcs

    def plot_img_stamps(self, fig, fig_dict, ra, dec, plot_rad_profile=True):

        hst_broad_band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)
        hst_ha_band = self.get_covered_hst_ha_band(ra=ra, dec=dec)
        nircam_band_list = self.get_covered_nircam_band_list(ra=ra, dec=dec)
        miri_band_list = self.get_covered_miri_band_list(ra=ra, dec=dec)

        band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)
        # check if H-alpha is available
        if ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_ha_cont_sub_target_name):
            if self.check_coords_covered_by_band(telescope="hst", ra=ra, dec=dec,
                                                 band=ObsTools.get_hst_ha_band(
                                                         target=self.phot_hst_ha_cont_sub_target_name),
                                                 max_dist_dist2hull_arcsec=2):
                band_list += [hst_ha_band]
                # check if Ha- continuum subtracted image is available
                if self.phot_hst_ha_cont_sub_target_name in obs_info.hst_ha_cont_sub_dict.keys():
                    band_list += [hst_ha_band + '_cont_sub']
        band_list += nircam_band_list + miri_band_list

        # load data
        self.load_obs_bands(band_list=band_list, flux_unit='MJy/sr', load_err=True, load_hst=True, load_hst_ha=True,
                               load_nircam=True, load_miri=True, load_astrosat=False)
        # load cutout stamps
        cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['stamp_size'],
                                                      band_list=band_list, include_err=True)

        hst_col_index = 0
        for hst_col_index, hst_stamp_band in enumerate(hst_broad_band_list):
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                    fig=fig,
                    left_align=fig_dict['rad_pro_left_align'],
                    bottom_align=fig_dict['rad_pro_bottom_align'],
                    width=fig_dict['rad_pro_width'],
                    height=fig_dict['rad_pro_height'],
                    space_vertical=fig_dict['rad_pro_space_vertical'],
                    space_horizontal=fig_dict['rad_pro_space_horizontal'],
                    row_idx=1, col_idx=hst_col_index)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=0.5,
                    img_err=cutout_dict_stamp['%s_err_cutout' % hst_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                    band=hst_stamp_band, instrument=ObsTools.get_hst_instrument(
                        target=self.phot_hst_target_name, band=hst_stamp_band))
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)
                ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray', alpha=0.7)
                ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                    psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                    linewidth=4, color='red')

                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                           labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(hst_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['hst_broad_band_color'])
                if hst_col_index == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['stamp_left_align'],
                bottom_align=fig_dict['stamp_bottom_align'],
                width=fig_dict['stamp_width'],
                height=fig_dict['stamp_height'],
                space_vertical=fig_dict['stamp_space_vertical'],
                space_horizontal=fig_dict['stamp_space_horizontal'],
                row_idx=1, col_idx=hst_col_index, projection=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs)
            norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data, norm=norm_hst_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                 pos=SkyCoord(ra=ra * u.deg,
                                                                              dec=dec * u.deg),
                                                                 wcs=cutout_dict_stamp[
                                                                     '%s_img_cutout' % hst_stamp_band].wcs,
                                                                 rad=0.3, hair_length=0.3,
                                                                 color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            if hst_col_index == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                    va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
            if hst_col_index == 1:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_2'], length_unit='pc',
                    phangs_target=self.phot_hst_target_name,
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                    va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

        # add h alpha
        if ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_target_name):
            if self.check_coords_covered_by_band(
                    telescope="hst", ra=ra, dec=dec,
                    band=ObsTools.get_hst_ha_band(target=self.phot_hst_target_name),
                    max_dist_dist2hull_arcsec=2):
                hst_ha_band = ObsTools.get_hst_ha_band(target=self.phot_hst_target_name)
                if plot_rad_profile:
                    ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['rad_pro_left_align'],
                        bottom_align=fig_dict['rad_pro_bottom_align'],
                        width=fig_dict['rad_pro_width'],
                        height=fig_dict['rad_pro_height'],
                        space_vertical=fig_dict['rad_pro_space_vertical'],
                        space_horizontal=fig_dict['rad_pro_space_horizontal'],
                        row_idx=1, col_idx=hst_col_index + 1)

                    radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                        img=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                        ra=ra, dec=dec, max_rad_arcsec=0.5,
                        img_err=cutout_dict_stamp['%s_err_cutout' % hst_ha_band].data)
                    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                        band=hst_ha_band, instrument=ObsTools.get_hst_instrument(
                            target=self.phot_hst_target_name, band=hst_ha_band))
                    ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray', alpha=0.7)
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                    mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4, color='red')

                    ax_rad_profile.set_yticklabels([])
                    ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                               labelsize=fig_dict['stamp_label_size'])
                    ax_rad_profile.set_title(hst_ha_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                             color=fig_dict['hst_ha_color'])
                    if hst_col_index == 0:
                        ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

                ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                    fig=fig,
                    left_align=fig_dict['stamp_left_align'],
                    bottom_align=fig_dict['stamp_bottom_align'],
                    width=fig_dict['stamp_width'],
                    height=fig_dict['stamp_height'],
                    space_vertical=fig_dict['stamp_space_vertical'],
                    space_horizontal=fig_dict['stamp_space_horizontal'],
                    row_idx=1, col_idx=hst_col_index + 1,
                    projection=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs)
                norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                    cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                    log_scale=True)
                ax_stamp.imshow(
                    cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data, norm=norm_hst_stamp,
                    cmap='Greys')
                plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                     pos=SkyCoord(ra=ra * u.deg,
                                                                                  dec=dec * u.deg),
                                                                     wcs=cutout_dict_stamp[
                                                                         '%s_img_cutout' % hst_ha_band].wcs,
                                                                     rad=0.3, hair_length=0.3,
                                                                     color='red', line_width=2)
                plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                                dec_tick_label=False, ra_axis_label=' ',
                                                                dec_axis_label=' ',
                                                                fontsize=fig_dict['stamp_label_size'],
                                                                labelsize=fig_dict['stamp_label_size'])
                if hst_col_index == 0:
                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data.shape,
                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                        bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                        bar_color='tab:red', text_color='tab:red',
                        line_width=4, fontsize=fig_dict['stamp_label_size'],
                        va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

                # same with continuum subtracted H-alpha
                hst_ha_cont_sub_band = hst_ha_band + '_cont_sub'
                if hst_ha_cont_sub_band in band_list:
                    # check if data is not corrupt
                    if (not (np.all(np.isnan(cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data)) |
                             np.all(cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data == 0))):

                        if plot_rad_profile:
                            ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                                fig=fig,
                                left_align=fig_dict['rad_pro_left_align'],
                                bottom_align=fig_dict['rad_pro_bottom_align'],
                                width=fig_dict['rad_pro_width'],
                                height=fig_dict['rad_pro_height'],
                                space_vertical=fig_dict['rad_pro_space_vertical'],
                                space_horizontal=fig_dict['rad_pro_space_horizontal'],
                                row_idx=1, col_idx=hst_col_index + 2)

                            radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                                img=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data,
                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                ra=ra, dec=dec, max_rad_arcsec=0.5,
                                img_err=cutout_dict_stamp['%s_err_cutout' % hst_ha_cont_sub_band].data)
                            psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                                band=hst_ha_band, instrument=ObsTools.get_hst_instrument(
                                    target=self.phot_hst_target_name, band=hst_ha_band))
                            ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray',
                                                        alpha=0.7)
                            ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                            mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                            ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                                psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                                linewidth=4, color='red')

                            ax_rad_profile.set_yticklabels([])
                            ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                                       labelsize=fig_dict['stamp_label_size'])
                            ax_rad_profile.set_title(hst_ha_cont_sub_band.upper(),
                                                     fontsize=fig_dict['stamp_title_font_size'],
                                                     color=fig_dict['hst_ha_color'])
                            if hst_col_index == 0:
                                ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

                        ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['stamp_left_align'],
                            bottom_align=fig_dict['stamp_bottom_align'],
                            width=fig_dict['stamp_width'],
                            height=fig_dict['stamp_height'],
                            space_vertical=fig_dict['stamp_space_vertical'],
                            space_horizontal=fig_dict['stamp_space_horizontal'],
                            row_idx=1, col_idx=hst_col_index + 2,
                            projection=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs)
                        norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                            cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data,
                            log_scale=True)
                        ax_stamp.imshow(
                            cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data, norm=norm_hst_stamp,
                            cmap='Greys')
                        plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                             pos=SkyCoord(ra=ra * u.deg,
                                                                                          dec=dec * u.deg),
                                                                             wcs=cutout_dict_stamp[
                                                                                 '%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                                                             rad=0.3, hair_length=0.3,
                                                                             color='red', line_width=2)
                        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                                        dec_tick_label=False, ra_axis_label=' ',
                                                                        dec_axis_label=' ',
                                                                        fontsize=fig_dict['stamp_label_size'],
                                                                        labelsize=fig_dict['stamp_label_size'])
                        if hst_col_index == 0:
                            plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                                ax=ax_stamp,
                                img_shape=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data.shape,
                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                                bar_color='tab:red', text_color='tab:red',
                                line_width=4, fontsize=fig_dict['stamp_label_size'],
                                va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

        nircam_col_index = 0
        for nircam_col_index, nircam_stamp_band in enumerate(nircam_band_list):
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                    fig=fig,
                    left_align=fig_dict['rad_pro_left_align'],
                    bottom_align=fig_dict['rad_pro_bottom_align'],
                    width=fig_dict['rad_pro_width'],
                    height=fig_dict['rad_pro_height'],
                    space_vertical=fig_dict['rad_pro_space_vertical'],
                    space_horizontal=fig_dict['rad_pro_space_horizontal'],
                    row_idx=0, col_idx=nircam_col_index)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=0.5,
                    img_err=cutout_dict_stamp['%s_err_cutout' % nircam_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=nircam_stamp_band, instrument='nircam')
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray', alpha=0.7)
                if nircam_col_index == 0:
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k', label='measured')

                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4, color='red', label='PSF')

                    ax_rad_profile.legend(frameon=False, fontsize=fig_dict['stamp_label_size'])
                else:
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4,
                                        color='red')
                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                           labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(nircam_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['nircam_color'])
                if nircam_col_index == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['stamp_left_align'],
                bottom_align=fig_dict['stamp_bottom_align'],
                width=fig_dict['stamp_width'],
                height=fig_dict['stamp_height'],
                space_vertical=fig_dict['stamp_space_vertical'],
                space_horizontal=fig_dict['stamp_space_horizontal'],
                row_idx=0, col_idx=nircam_col_index,
                projection=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs)
            norm_nircam_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data, norm=norm_nircam_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                 pos=SkyCoord(ra=ra * u.deg,
                                                                              dec=dec * u.deg),
                                                                 wcs=cutout_dict_stamp[
                                                                     '%s_img_cutout' % nircam_stamp_band].wcs,
                                                                 rad=0.3, hair_length=0.3,
                                                                 color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            if nircam_col_index == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                    va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

        miri_col_index = 0
        for miri_col_index, miri_stamp_band in enumerate(miri_band_list):
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                    fig=fig,
                    left_align=fig_dict['rad_pro_left_align'],
                    bottom_align=fig_dict['rad_pro_bottom_align'],
                    width=fig_dict['rad_pro_width'],
                    height=fig_dict['rad_pro_height'],
                    space_vertical=fig_dict['rad_pro_space_vertical'],
                    space_horizontal=fig_dict['rad_pro_space_horizontal'],
                    row_idx=0, col_idx=nircam_col_index + miri_col_index + 1)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=1.0,
                    img_err=cutout_dict_stamp['%s_err_cutout' % miri_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=miri_stamp_band, instrument='miri')
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray', alpha=0.7)
                ax_rad_profile.plot(radius, profile, linewidth=4, color='k')

                ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                    psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                    linewidth=4, color='red')

                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                           labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(miri_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['miri_color'])
                if miri_col_index == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['stamp_left_align'],
                bottom_align=fig_dict['stamp_bottom_align'],
                width=fig_dict['stamp_width'],
                height=fig_dict['stamp_height'],
                space_vertical=fig_dict['stamp_space_vertical'],
                space_horizontal=fig_dict['stamp_space_horizontal'],
                row_idx=0, col_idx=nircam_col_index + miri_col_index + 1,
                projection=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs)
            norm_miri_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data, norm=norm_miri_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                 pos=SkyCoord(ra=ra * u.deg,
                                                                              dec=dec * u.deg),
                                                                 wcs=cutout_dict_stamp[
                                                                     '%s_img_cutout' % miri_stamp_band].wcs,
                                                                 rad=0.3, hair_length=0.3,
                                                                 color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            if miri_col_index == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                    va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

    def plot_img_stamps_all(self, fig, fig_dict, ra, dec, plot_rad_profile=True, individual_band_list=None):

        prelim_hst_broad_band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec, max_dist_dist2hull_arcsec=0.5)
        prelim_hst_ha_band = self.get_covered_hst_ha_band(ra=ra, dec=dec, max_dist_dist2hull_arcsec=0.5)
        prelim_nircam_band_list = self.get_covered_nircam_band_list(ra=ra, dec=dec, max_dist_dist2hull_arcsec=0.5)
        prelim_miri_band_list = self.get_covered_miri_band_list(ra=ra, dec=dec, max_dist_dist2hull_arcsec=0.5)

        prelim_band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec, max_dist_dist2hull_arcsec=0.5)

        # check if H-alpha is available
        if ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_ha_cont_sub_target_name):
            if self.check_coords_covered_by_band(obs="hst", ra=ra, dec=dec,
                                                 band=ObsTools.get_hst_ha_band(
                                                         target=self.phot_hst_ha_cont_sub_target_name),
                                                 max_dist_dist2hull_arcsec=0.5):
                prelim_band_list += [prelim_hst_ha_band]
                # check if Ha- continuum subtracted image is available
                if self.phot_hst_ha_cont_sub_target_name in obs_info.hst_ha_cont_sub_dict.keys():
                    prelim_band_list += [prelim_hst_ha_band + '_cont_sub']
        prelim_band_list += prelim_nircam_band_list + prelim_miri_band_list

        if individual_band_list is None:
            band_list = prelim_band_list
            hst_broad_band_list = prelim_hst_broad_band_list
            hst_ha_band = prelim_hst_ha_band
            nircam_band_list = prelim_nircam_band_list
            miri_band_list = prelim_miri_band_list
        else:
            band_list = []
            hst_broad_band_list = []
            hst_ha_band = []
            nircam_band_list = []
            miri_band_list = []
            for band in prelim_band_list:
                if band in individual_band_list:
                    band_list.append(band)
                    if band in prelim_hst_broad_band_list:
                        hst_broad_band_list.append(band)

                    if band in prelim_hst_ha_band:
                        hst_ha_band.append(band)

                    if band in prelim_nircam_band_list:
                        nircam_band_list.append(band)
                    if band in prelim_miri_band_list:
                        miri_band_list.append(band)

        print(band_list)
        # load data
        self.load_obs_bands(band_list=band_list, flux_unit='MJy/sr', load_err=True, load_hst=True, load_hst_ha=True,
                               load_nircam=True, load_miri=True, load_astrosat=False)

        # load cutout stamps
        cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['stamp_size'],
                                                      band_list=band_list, include_err=True)

        col_idx = 0
        max_n_cols = 9
        running_row_idx = np.ceil(len(band_list) / max_n_cols)

        for hst_stamp_band in hst_broad_band_list:
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                    fig=fig,
                    left_align=fig_dict['rad_pro_left_align'],
                    bottom_align=fig_dict['rad_pro_bottom_align'],
                    width=fig_dict['rad_pro_width'],
                    height=fig_dict['rad_pro_height'],
                    space_vertical=fig_dict['rad_pro_space_vertical'],
                    space_horizontal=fig_dict['rad_pro_space_horizontal'],
                    row_idx=running_row_idx, col_idx=col_idx)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=0.5,
                    img_err=cutout_dict_stamp['%s_err_cutout' % hst_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                    band=hst_stamp_band, instrument=ObsTools.get_hst_instrument(
                        target=self.phot_hst_target_name, band=hst_stamp_band))
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)
                ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray', alpha=0.7)
                ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                    psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                    linewidth=4, color='red')

                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                           labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(hst_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['hst_broad_band_color'])
                if col_idx == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['stamp_left_align'],
                bottom_align=fig_dict['stamp_bottom_align'],
                width=fig_dict['stamp_width'],
                height=fig_dict['stamp_height'],
                space_vertical=fig_dict['stamp_space_vertical'],
                space_horizontal=fig_dict['stamp_space_horizontal'],
                row_idx=running_row_idx, col_idx=col_idx,
                projection=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs)
            norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data, norm=norm_hst_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                 pos=SkyCoord(ra=ra * u.deg,
                                                                              dec=dec * u.deg),
                                                                 wcs=cutout_dict_stamp[
                                                                     '%s_img_cutout' % hst_stamp_band].wcs,
                                                                 rad=0.3, hair_length=0.3,
                                                                 color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            if col_idx == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                    va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
            if col_idx == 1:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % hst_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_2'], length_unit='pc',
                    phangs_target=self.phot_hst_target_name,
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                    va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
            col_idx += 1
            if col_idx >= max_n_cols:
                col_idx = 0
                running_row_idx -= 1

        # add h alpha
        if ObsTools.check_hst_ha_cont_sub_obs(target=self.phot_hst_target_name):
            if self.check_coords_covered_by_band(
                    obs="hst", ra=ra, dec=dec,
                    band=ObsTools.get_hst_ha_band(target=self.phot_hst_target_name),
                    max_dist_dist2hull_arcsec=0.5):
                hst_ha_band = ObsTools.get_hst_ha_band(target=self.phot_hst_target_name)
                if plot_rad_profile:
                    ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                        fig=fig,
                        left_align=fig_dict['rad_pro_left_align'],
                        bottom_align=fig_dict['rad_pro_bottom_align'],
                        width=fig_dict['rad_pro_width'],
                        height=fig_dict['rad_pro_height'],
                        space_vertical=fig_dict['rad_pro_space_vertical'],
                        space_horizontal=fig_dict['rad_pro_space_horizontal'],
                        row_idx=running_row_idx, col_idx=col_idx)

                    radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                        img=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                        ra=ra, dec=dec, max_rad_arcsec=0.5,
                        img_err=cutout_dict_stamp['%s_err_cutout' % hst_ha_band].data)
                    psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                        band=hst_ha_band, instrument=ObsTools.get_hst_instrument(
                            target=self.phot_hst_target_name, band=hst_ha_band))
                    ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray', alpha=0.7)
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                    mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4, color='red')

                    ax_rad_profile.set_yticklabels([])
                    ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                               labelsize=fig_dict['stamp_label_size'])
                    ax_rad_profile.set_title(hst_ha_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                             color=fig_dict['hst_ha_color'])
                    if col_idx == 0:
                        ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

                ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                    fig=fig,
                    left_align=fig_dict['stamp_left_align'],
                    bottom_align=fig_dict['stamp_bottom_align'],
                    width=fig_dict['stamp_width'],
                    height=fig_dict['stamp_height'],
                    space_vertical=fig_dict['stamp_space_vertical'],
                    space_horizontal=fig_dict['stamp_space_horizontal'],
                    row_idx=running_row_idx, col_idx=col_idx,
                    projection=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs)
                norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                    cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data,
                    log_scale=True)
                ax_stamp.imshow(
                    cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data, norm=norm_hst_stamp,
                    cmap='Greys')
                plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                     pos=SkyCoord(ra=ra * u.deg,
                                                                                  dec=dec * u.deg),
                                                                     wcs=cutout_dict_stamp[
                                                                         '%s_img_cutout' % hst_ha_band].wcs,
                                                                     rad=0.3, hair_length=0.3,
                                                                     color='red', line_width=2)
                plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                                dec_tick_label=False, ra_axis_label=' ',
                                                                dec_axis_label=' ',
                                                                fontsize=fig_dict['stamp_label_size'],
                                                                labelsize=fig_dict['stamp_label_size'])
                if col_idx == 0:
                    plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                        ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].data.shape,
                        wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_band].wcs,
                        bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                        bar_color='tab:red', text_color='tab:red',
                        line_width=4, fontsize=fig_dict['stamp_label_size'],
                        va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
                col_idx += 1
                if col_idx >= max_n_cols:
                    col_idx = 0
                    running_row_idx -= 1

                # same with continuum subtracted H-alpha
                hst_ha_cont_sub_band = hst_ha_band + '_cont_sub'
                if hst_ha_cont_sub_band in band_list:
                    # check if data is not corrupt
                    if (not (np.all(np.isnan(cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data)) |
                             np.all(cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data == 0))):

                        if plot_rad_profile:
                            ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                                fig=fig,
                                left_align=fig_dict['rad_pro_left_align'],
                                bottom_align=fig_dict['rad_pro_bottom_align'],
                                width=fig_dict['rad_pro_width'],
                                height=fig_dict['rad_pro_height'],
                                space_vertical=fig_dict['rad_pro_space_vertical'],
                                space_horizontal=fig_dict['rad_pro_space_horizontal'],
                                row_idx=running_row_idx, col_idx=col_idx)

                            radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                                img=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data,
                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                ra=ra, dec=dec, max_rad_arcsec=0.5,
                                img_err=cutout_dict_stamp['%s_err_cutout' % hst_ha_cont_sub_band].data)
                            psf_dict = phot_tools.PSFTools.load_obs_psf_dict(
                                band=hst_ha_band, instrument=ObsTools.get_hst_instrument(
                                    target=self.phot_hst_target_name, band=hst_ha_band))
                            ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray',
                                                        alpha=0.7)
                            ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                            mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                            ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                                psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                                linewidth=4, color='red')

                            ax_rad_profile.set_yticklabels([])
                            ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                                       labelsize=fig_dict['stamp_label_size'])
                            ax_rad_profile.set_title(hst_ha_cont_sub_band.upper(),
                                                     fontsize=fig_dict['stamp_title_font_size'],
                                                     color=fig_dict['hst_ha_color'])
                            if col_idx == 0:
                                ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

                        ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                            fig=fig,
                            left_align=fig_dict['stamp_left_align'],
                            bottom_align=fig_dict['stamp_bottom_align'],
                            width=fig_dict['stamp_width'],
                            height=fig_dict['stamp_height'],
                            space_vertical=fig_dict['stamp_space_vertical'],
                            space_horizontal=fig_dict['stamp_space_horizontal'],
                            row_idx=running_row_idx, col_idx=col_idx,
                            projection=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs)
                        norm_hst_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                            cutout_list=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data,
                            log_scale=True)
                        ax_stamp.imshow(
                            cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data, norm=norm_hst_stamp,
                            cmap='Greys')
                        plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                             pos=SkyCoord(ra=ra * u.deg,
                                                                                          dec=dec * u.deg),
                                                                             wcs=cutout_dict_stamp[
                                                                                 '%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                                                             rad=0.3, hair_length=0.3,
                                                                             color='red', line_width=2)
                        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                                        dec_tick_label=False, ra_axis_label=' ',
                                                                        dec_axis_label=' ',
                                                                        fontsize=fig_dict['stamp_label_size'],
                                                                        labelsize=fig_dict['stamp_label_size'])
                        if col_idx == 0:
                            plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                                ax=ax_stamp,
                                img_shape=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].data.shape,
                                wcs=cutout_dict_stamp['%s_img_cutout' % hst_ha_cont_sub_band].wcs,
                                bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                                bar_color='tab:red', text_color='tab:red',
                                line_width=4, fontsize=fig_dict['stamp_label_size'],
                                va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

                        col_idx += 1
                        if col_idx >= max_n_cols:
                            col_idx = 0
                            running_row_idx -= 1

        for nircam_stamp_band in nircam_band_list:
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                    fig=fig,
                    left_align=fig_dict['rad_pro_left_align'],
                    bottom_align=fig_dict['rad_pro_bottom_align'],
                    width=fig_dict['rad_pro_width'],
                    height=fig_dict['rad_pro_height'],
                    space_vertical=fig_dict['rad_pro_space_vertical'],
                    space_horizontal=fig_dict['rad_pro_space_horizontal'],
                    row_idx=running_row_idx, col_idx=col_idx)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=0.5,
                    img_err=cutout_dict_stamp['%s_err_cutout' % nircam_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=nircam_stamp_band, instrument='nircam')
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray', alpha=0.7)
                if col_idx == 0:
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k', label='measured')

                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4, color='red', label='PSF')

                    ax_rad_profile.legend(frameon=False, fontsize=fig_dict['stamp_label_size'])
                else:
                    ax_rad_profile.plot(radius, profile, linewidth=4, color='k')
                    ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                        psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                        linewidth=4,
                                        color='red')
                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                           labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(nircam_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['nircam_color'])
                if col_idx == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['stamp_left_align'],
                bottom_align=fig_dict['stamp_bottom_align'],
                width=fig_dict['stamp_width'],
                height=fig_dict['stamp_height'],
                space_vertical=fig_dict['stamp_space_vertical'],
                space_horizontal=fig_dict['stamp_space_horizontal'],
                row_idx=running_row_idx, col_idx=col_idx,
                projection=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs)
            norm_nircam_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data, norm=norm_nircam_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                 pos=SkyCoord(ra=ra * u.deg,
                                                                              dec=dec * u.deg),
                                                                 wcs=cutout_dict_stamp[
                                                                     '%s_img_cutout' % nircam_stamp_band].wcs,
                                                                 rad=0.3, hair_length=0.3,
                                                                 color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            if col_idx == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % nircam_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                    va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

            col_idx += 1
            if col_idx >= max_n_cols:
                col_idx = 0
                running_row_idx -= 1

        for miri_stamp_band in miri_band_list:
            if plot_rad_profile:
                ax_rad_profile = plotting_tools.AxisTools.add_panel_axis(
                    fig=fig,
                    left_align=fig_dict['rad_pro_left_align'],
                    bottom_align=fig_dict['rad_pro_bottom_align'],
                    width=fig_dict['rad_pro_width'],
                    height=fig_dict['rad_pro_height'],
                    space_vertical=fig_dict['rad_pro_space_vertical'],
                    space_horizontal=fig_dict['rad_pro_space_horizontal'],
                    row_idx=running_row_idx, col_idx=col_idx)

                radius, profile, error = phot_tools.ProfileTools.get_rad_profile_from_img(
                    img=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data,
                    wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                    ra=ra, dec=dec, max_rad_arcsec=1.0,
                    img_err=cutout_dict_stamp['%s_err_cutout' % miri_stamp_band].data)

                psf_dict = phot_tools.PSFTools.load_obs_psf_dict(band=miri_stamp_band, instrument='miri')
                mask_psf_rad = psf_dict['radius_arcsec'] < np.max(radius)

                ax_rad_profile.fill_between(radius, profile - error, profile + error, color='gray', alpha=0.7)
                ax_rad_profile.plot(radius, profile, linewidth=4, color='k')

                ax_rad_profile.plot(psf_dict['radius_arcsec'][mask_psf_rad],
                                    psf_dict['psf_profile'][mask_psf_rad] / np.max(psf_dict['psf_profile']),
                                    linewidth=4, color='red')

                ax_rad_profile.set_yticklabels([])
                ax_rad_profile.tick_params(axis='both', which='both', width=2, direction='in',
                                           labelsize=fig_dict['stamp_label_size'])
                ax_rad_profile.set_title(miri_stamp_band.upper(), fontsize=fig_dict['stamp_title_font_size'],
                                         color=fig_dict['miri_color'])
                if col_idx == 0:
                    ax_rad_profile.set_xlabel('rad. [\"]', fontsize=fig_dict['stamp_label_size'])

            ax_stamp = plotting_tools.AxisTools.add_panel_axis(
                fig=fig,
                left_align=fig_dict['stamp_left_align'],
                bottom_align=fig_dict['stamp_bottom_align'],
                width=fig_dict['stamp_width'],
                height=fig_dict['stamp_height'],
                space_vertical=fig_dict['stamp_space_vertical'],
                space_horizontal=fig_dict['stamp_space_horizontal'],
                row_idx=running_row_idx, col_idx=col_idx,
                projection=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs)
            norm_miri_stamp = plotting_tools.ColorBarTools.compute_cbar_norm(
                cutout_list=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data,
                log_scale=True)
            ax_stamp.imshow(
                cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data, norm=norm_miri_stamp,
                cmap='Greys')
            plotting_tools.WCSPlottingTools.plot_coord_crosshair(ax=ax_stamp,
                                                                 pos=SkyCoord(ra=ra * u.deg,
                                                                              dec=dec * u.deg),
                                                                 wcs=cutout_dict_stamp[
                                                                     '%s_img_cutout' % miri_stamp_band].wcs,
                                                                 rad=0.3, hair_length=0.3,
                                                                 color='red', line_width=2)
            plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_stamp, ra_tick_label=False,
                                                            dec_tick_label=False, ra_axis_label=' ', dec_axis_label=' ',
                                                            fontsize=fig_dict['stamp_label_size'],
                                                            labelsize=fig_dict['stamp_label_size'])
            if col_idx == 0:
                plotting_tools.WCSPlottingTools.plot_img_scale_bar(
                    ax=ax_stamp, img_shape=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].data.shape,
                    wcs=cutout_dict_stamp['%s_img_cutout' % miri_stamp_band].wcs,
                    bar_length=fig_dict['stamp_scale_bar_length_1'], length_unit='arcsec',
                    bar_color='tab:red', text_color='tab:red',
                    line_width=4, fontsize=fig_dict['stamp_label_size'],
                    va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)

            col_idx += 1
            if col_idx >= max_n_cols:
                col_idx = 0
                running_row_idx -= 1

    def plot_sed_panel(self, fig, fig_dict, ra, dec, individual_band_list=None):

        prelim_hst_broad_band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)
        prelim_hst_ha_band = self.get_covered_hst_ha_band(ra=ra, dec=dec)
        prelim_nircam_band_list = self.get_covered_nircam_band_list(ra=ra, dec=dec)
        prelim_miri_band_list = self.get_covered_miri_band_list(ra=ra, dec=dec)

        print('prelim_hst_broad_band_list ', prelim_hst_broad_band_list)
        print('prelim_hst_ha_band ', prelim_hst_ha_band)
        print('prelim_nircam_band_list ', prelim_nircam_band_list)
        print('prelim_miri_band_list ', prelim_miri_band_list)

        prelim_band_list = self.get_covered_hst_broad_band_list(ra=ra, dec=dec)
        if ObsTools.check_hst_ha_obs(target=self.phot_hst_target_name):
            if self.check_coords_covered_by_band(obs="hst", ra=ra, dec=dec,
                                                 band=ObsTools.get_hst_ha_band(
                                                         target=self.phot_hst_target_name),
                                                 max_dist_dist2hull_arcsec=2):
                prelim_band_list += [prelim_hst_ha_band]
                # check if Ha- continuum subtracted image is available
                if self.phot_hst_ha_cont_sub_target_name in obs_info.hst_ha_cont_sub_dict.keys():
                    prelim_band_list += [prelim_hst_ha_band + '_cont_sub']
        prelim_band_list += prelim_nircam_band_list + prelim_miri_band_list

        if individual_band_list is None:
            band_list = prelim_band_list
            hst_broad_band_list = prelim_hst_broad_band_list
            hst_ha_band = prelim_hst_ha_band
            nircam_band_list = prelim_nircam_band_list
            miri_band_list = prelim_miri_band_list
        else:
            band_list = []
            hst_broad_band_list = []
            hst_ha_band = None
            nircam_band_list = []
            miri_band_list = []
            for band in prelim_band_list:
                if band in individual_band_list:
                    band_list.append(band)
                    if band in prelim_hst_broad_band_list:
                        hst_broad_band_list.append(band)

                    if band in prelim_hst_ha_band:
                        hst_ha_band = band

                    if band in prelim_nircam_band_list:
                        nircam_band_list.append(band)
                    if band in prelim_miri_band_list:
                        miri_band_list.append(band)

        # load data in case it is not yet loaded
        self.load_obs_bands(band_list=band_list, flux_unit='mJy', load_err=True, load_hst=True, load_hst_ha=True,
                               load_nircam=True, load_miri=True, load_astrosat=False)
        # load large cutout dict for flux density estimation
        self.change_phangs_band_units(band_list=band_list, new_unit='mJy')

        # # load cutout stamps
        # cutout_dict_stamp = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['sed_size'],
        #                                               band_list=band_list, include_err=True)

        # add sed axis
        ax_sed = fig.add_axes([
            fig_dict['sed_left_align'],
            fig_dict['sed_bottom_align'],
            fig_dict['sed_width'],
            fig_dict['sed_height'],
        ])

        for band in hst_broad_band_list:
            morph_flux_dict = self.compute_morph_phot(
                ra=ra, dec=dec, band=band,
                instrument=ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band),
                cutout_size=fig_dict['sed_size_hst'])

            mean_wave = ObsTools.get_hst_band_wave(
                band=band,
                instrument=ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band))
            min_wave = ObsTools.get_hst_band_wave(
                band=band,
                instrument=ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band),
                wave_estimator='min_wave')
            max_wave = ObsTools.get_hst_band_wave(
                band=band,
                instrument=ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=band),
                wave_estimator='max_wave')

            ax_sed.errorbar(mean_wave, morph_flux_dict['flux'],
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=morph_flux_dict['flux_err'],
                            fmt='v', color=fig_dict['hst_broad_band_color'], ms=20)

        if hst_ha_band:
            morph_flux_dict = self.compute_morph_phot(
                ra=ra, dec=dec, band=hst_ha_band,
                instrument=ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=hst_ha_band),
                cutout_size=fig_dict['sed_size_hst'])

            mean_wave = ObsTools.get_hst_band_wave(
                band=hst_ha_band,
                instrument=ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=hst_ha_band))
            min_wave = ObsTools.get_hst_band_wave(
                band=hst_ha_band,
                instrument=ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=hst_ha_band),
                wave_estimator='min_wave')
            max_wave = ObsTools.get_hst_band_wave(
                band=hst_ha_band,
                instrument=ObsTools.get_hst_instrument(target=self.phot_hst_target_name, band=hst_ha_band),
                wave_estimator='max_wave')

            ax_sed.errorbar(mean_wave, morph_flux_dict['flux'],
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=morph_flux_dict['flux_err'],
                            fmt='v', color=fig_dict['hst_ha_color'], ms=20)

        for band in nircam_band_list:
            morph_flux_dict = self.compute_morph_phot(
                ra=ra, dec=dec, band=band,
                instrument='nircam',
                cutout_size=fig_dict['sed_size_nircam'])

            mean_wave = ObsTools.get_jwst_band_wave(
                band=band,
                instrument='nircam')
            min_wave = ObsTools.get_jwst_band_wave(
                band=band,
                instrument='nircam',
                wave_estimator='min_wave')
            max_wave = ObsTools.get_jwst_band_wave(
                band=band,
                instrument='nircam',
                wave_estimator='max_wave')

            ax_sed.errorbar(mean_wave, morph_flux_dict['flux'],
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=morph_flux_dict['flux_err'],
                            fmt='v', color=fig_dict['nircam_color'], ms=20)

        for band in miri_band_list:
            morph_flux_dict = self.compute_morph_phot(
                ra=ra, dec=dec, band=band,
                instrument='miri',
                cutout_size=fig_dict['sed_size_miri'])

            mean_wave = ObsTools.get_jwst_band_wave(
                band=band,
                instrument='miri')
            min_wave = ObsTools.get_jwst_band_wave(
                band=band,
                instrument='miri',
                wave_estimator='min_wave')
            max_wave = ObsTools.get_jwst_band_wave(
                band=band,
                instrument='miri',
                wave_estimator='max_wave')

            ax_sed.errorbar(mean_wave, morph_flux_dict['flux'],
                            xerr=[[mean_wave - min_wave], [max_wave - mean_wave]],
                            yerr=morph_flux_dict['flux_err'],
                            fmt='v', color=fig_dict['miri_color'], ms=20)

        # add cluster catalog cross-match
        if self.phot_hst_target_name in obs_info.hst_cluster_cat_target_list:
            phangs_cluster = ClusterCatAccess()
            cross_match_map = phangs_cluster.get_hst_cc_cross_match_mask(target=self.phot_hst_target_name,
                                                                         ra=ra, dec=dec, toleance_arcsec=0.1,
                                                                         classify='human', cluster_class='class12')
            if sum(cross_match_map) > 0:
                sed_age = phangs_cluster.get_hst_cc_age(target=self.phot_hst_target_name)[cross_match_map]
                sed_mstar = phangs_cluster.get_hst_cc_mstar(target=self.phot_hst_target_name)[cross_match_map]
                sed_ebv = phangs_cluster.get_hst_cc_ebv(target=self.phot_hst_target_name)[cross_match_map]
                age_string = plotting_tools.StrTools.age2label(age=sed_age[0])
                mstar_string = plotting_tools.StrTools.mstar2label(mstar=sed_mstar)
                muse_title_str = (r'HST-SED fit params'
                                  + '\n' +
                                  r'age = ' + age_string
                                  + '\n' +
                                  r'M$_{*}$ = ' + mstar_string
                                  + '\n' +
                                  r'E(B-V) = %.2f mag' % sed_ebv)
                t = ax_sed.text(1.03, 0.93, muse_title_str, horizontalalignment='left', verticalalignment='top',
                                transform=ax_sed.transAxes, fontsize=fig_dict['sed_title_font_size'])
                t.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))

        ax_sed.set_yscale('log')
        ax_sed.set_xscale('log')
        ax_sed.set_xlabel(r'Wavelength [$\mu$m]', fontsize=fig_dict['sed_label_size'])
        ax_sed.set_ylabel(r'Flux [mJy]', fontsize=fig_dict['sed_label_size'])
        ax_sed.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fig_dict['sed_label_size'])

    def plot_muse_spec(self, fig, fig_dict, ra, dec, ppxf_fit_dict):

        if not self.check_coords_covered_by_muse(ra=ra, dec=dec, res='copt', max_dist_dist2hull_arcsec=2):
            return None

        # add sed axis
        ax_muse_spec = fig.add_axes([
            fig_dict['muse_spec_left_align'],
            fig_dict['muse_spec_bottom_align'],
            fig_dict['muse_spec_width'],
            fig_dict['muse_spec_height'],
        ])

        ax_muse_spec.plot(ppxf_fit_dict['wave'], ppxf_fit_dict['total_flux'] * 1e16, linewidth=3, color='k',
                          label='Obs. Spectrum')
        ax_muse_spec.plot(ppxf_fit_dict['wave'], ppxf_fit_dict['best_fit'] * 1e16, linewidth=3, color='tab:red',
                          label='Best total fit')
        ax_muse_spec.plot(ppxf_fit_dict['wave'], ppxf_fit_dict['continuum_best_fit'] * 1e16, linewidth=3,
                          color='tab:orange', label='Stellar Continuum fit')
        # plt.rc('text', usetex=True)
        # muse_title_str = (r'$\underline{\rm PPxF\,fit\,parameters}$'
        muse_title_str = (r'${\rm PPxF\,fit\,parameters}$'
                          + '\n' +
                          r'age = %.2f Myr ' % (10 ** (ppxf_fit_dict['ages']) * 1e-6)
                          + '\n' +
                          r'[M/H] = %.3f' % ppxf_fit_dict['met']
                          + '\n' +
                          r'A$_{\rm v} (star) = $%.2f mag' % (ppxf_fit_dict['star_red'])
                          + '\n' +
                          r'A$_{\rm v} (gas) = $%.2f mag' % (ppxf_fit_dict['gas_red']))
        t = ax_muse_spec.text(1.02, 0.05, muse_title_str, horizontalalignment='left', verticalalignment='bottom',
                              transform=ax_muse_spec.transAxes, fontsize=fig_dict['muse_spec_title_font_size'])
        t.set_bbox(dict(facecolor='grey', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))

        ax_muse_spec.tick_params(axis='both', which='both', width=2, direction='in',
                                 labelsize=fig_dict['muse_spec_label_size'])
        ax_muse_spec.set_xlabel(r'Wavelength [${\rm \AA}$]', fontsize=fig_dict['muse_spec_label_size'])
        ax_muse_spec.set_ylabel(r'$\phi$ [10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$]',
                                fontsize=fig_dict['muse_spec_label_size'])

        ax_muse_spec.legend(frameon=False, fontsize=fig_dict['muse_spec_label_size'])

        # get x limits
        if fig_dict['muse_spec_x_lim'][0] == 'min':
            wave_min = np.nanmin(ppxf_fit_dict['wave'])
        else:
            wave_min = fig_dict['muse_spec_x_lim'][0]

        if fig_dict['muse_spec_x_lim'][1] == 'max':
            wave_max = np.nanmax(ppxf_fit_dict['wave'])
        else:
            wave_max = fig_dict['muse_spec_x_lim'][1]

        # get y-limits
        mask_displayed_wave = (ppxf_fit_dict['wave'] > wave_min) & (ppxf_fit_dict['wave'] < wave_max)
        if isinstance(fig_dict['muse_spec_y_lim'], str):
            if fig_dict['muse_spec_y_lim'] == 'cont':
                flux_min = np.nanmin(ppxf_fit_dict['continuum_best_fit'][mask_displayed_wave] * 1e16)
                flux_max = np.nanmax(ppxf_fit_dict['continuum_best_fit'][mask_displayed_wave] * 1e16)
                flux_lim_lo = flux_min - (flux_max - flux_min) * 0.05
                flux_lim_hi = flux_max + (flux_max - flux_min) * 0.05
            elif fig_dict['muse_spec_y_lim'] == 'total':
                flux_min = np.nanmin(ppxf_fit_dict['total_flux'][mask_displayed_wave] * 1e16)
                flux_max = np.nanmax(ppxf_fit_dict['total_flux'][mask_displayed_wave] * 1e16)
                flux_lim_lo = flux_min - (flux_max - flux_min) * 0.05
                flux_lim_hi = flux_max + (flux_max - flux_min) * 0.05
            else:
                flux_lim_lo = None
                flux_lim_hi = None
        elif isinstance(fig_dict['muse_spec_y_lim'], tuple):
            flux_lim_lo, flux_lim_hi = fig_dict['muse_spec_y_lim']
        else:
            flux_lim_lo = None
            flux_lim_hi = None

        # set x-labels
        ax_muse_spec.set_xlim(wave_min, wave_max)
        ax_muse_spec.set_ylim(flux_lim_lo, flux_lim_hi)

        # plot ha plot cutout
        muse_dap_map_cutout_dict = self.get_muse_dap_map_cutout(
            ra_cutout=ra, dec_cutout=dec, cutout_size=fig_dict['muse_ha_zoom_in_size'],
            map_type_list=fig_dict['muse_ha_zoom_in_map_typ'], res=fig_dict['muse_ha_zoom_in_res'],
            ssp_model=fig_dict['muse_ha_zoom_in_ssp_model'])
        dap_data_identifier = helper_func.SpecHelper.get_dap_data_identifier(res=fig_dict['muse_ha_zoom_in_res'],
                                                                             ssp_model=fig_dict[
                                                                                 'muse_ha_zoom_in_ssp_model'])
        muse_ha_cutout = muse_dap_map_cutout_dict[
            '%s_%s_img_cutout' % (dap_data_identifier, fig_dict['muse_ha_zoom_in_map_typ'])]

        # add axis
        ax_muse_ha_zoom_in = fig.add_axes([
            fig_dict['muse_ha_zoom_in_left_align'],
            fig_dict['muse_ha_zoom_in_bottom_align'],
            fig_dict['muse_ha_zoom_in_width'],
            fig_dict['muse_ha_zoom_in_height']], projection=muse_ha_cutout.wcs)

        norm_muse_ha = plotting_tools.ColorBarTools.compute_cbar_norm(cutout_list=muse_ha_cutout.data, log_scale=True)
        ax_muse_ha_zoom_in.imshow(muse_ha_cutout.data, norm=norm_muse_ha, cmap='cividis')
        ax_muse_ha_zoom_in.set_title(r'MUSE H$\alpha$', fontsize=fig_dict['muse_spec_title_font_size'], color='k')
        plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_muse_ha_zoom_in,
                                                          pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                          rad=ppxf_fit_dict['rad_arcsec'],
                                                          color='cyan', line_width=4)
        plotting_tools.WCSPlottingTools.plot_coord_circle(ax=ax_muse_ha_zoom_in,
                                                          pos=SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                                                          rad=ppxf_fit_dict['rad_arcsec'],
                                                          color='k', line_width=4, line_style='--')
        ax_muse_ha_zoom_in.plot([], [], color='cyan', linewidth=4, label='Spec. apert.')
        ax_muse_ha_zoom_in.plot([], [], color='k', linewidth=4, linestyle='--', label='PSF FWHM')
        ax_muse_ha_zoom_in.legend(frameon=False, bbox_to_anchor=[0.95, 0.3], fontsize=fig_dict['muse_spec_label_size'])

        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
            ax=ax_muse_ha_zoom_in, img_shape=muse_ha_cutout.data.shape,
            wcs=muse_ha_cutout.wcs,
            bar_length=fig_dict['muse_scale_bar_length_1'], length_unit='arcsec',
            bar_color='tab:red', text_color='tab:red',
            line_width=4, fontsize=fig_dict['muse_spec_label_size'],
            va='bottom', ha='left', x_offset=0.05, y_offset=0.05, text_y_offset_diff=0.01)
        plotting_tools.WCSPlottingTools.plot_img_scale_bar(
            ax=ax_muse_ha_zoom_in, img_shape=muse_ha_cutout.data.shape,
            wcs=muse_ha_cutout.wcs,
            bar_length=fig_dict['muse_scale_bar_length_2'], length_unit='pc',
            phangs_target=self.phot_target_name,
            bar_color='tab:red', text_color='tab:red',
            line_width=4, fontsize=fig_dict['muse_spec_label_size'],
            va='top', ha='right', x_offset=0.1, y_offset=0.15, text_y_offset_diff=0.01)
        plotting_tools.WCSPlottingTools.arr_axis_params(ax=ax_muse_ha_zoom_in, ra_tick_label=False,
                                                        dec_tick_label=False,
                                                        ra_axis_label=' ', dec_axis_label=' ',
                                                        fontsize=fig_dict['muse_spec_label_size'],
                                                        labelsize=fig_dict['muse_spec_label_size'])
        return None

    def plot_spec_features(self, fig, fig_dict, em_line_fit_dict):
        # add sed axis

        if ((4863 in em_line_fit_dict['ln_list']) & (4960 in em_line_fit_dict['ln_list']) &
                (5008 in em_line_fit_dict['ln_list'])):
            ax_hb_oiii_spec = fig.add_axes([
                fig_dict['hb_oiii_left_align'],
                fig_dict['hb_oiii_bottom_align'],
                fig_dict['hb_oiii_width'],
                fig_dict['hb_oiii_height'],
            ])
            ax_hb_oiii_res_spec = fig.add_axes([
                fig_dict['hb_oiii_res_left_align'],
                fig_dict['hb_oiii_res_bottom_align'],
                fig_dict['hb_oiii_res_width'],
                fig_dict['hb_oiii_res_height'],
            ])

            plotting_tools.SpecPlotTools.plot_em_line_spec(ax=ax_hb_oiii_spec, em_line_fit_dict=em_line_fit_dict,
                                                           line_list=[4863, 4960, 5008],
                                                           left_offset=fig_dict['hb_oiii_left_lim_offst'],
                                                           right_offset=fig_dict['hb_oiii_right_lim_offst'],
                                                           display_legend=True,
                                                           display_x_label=False,
                                                           ax_res=ax_hb_oiii_res_spec,
                                                           font_size_label=fig_dict['em_label_size'],
                                                           font_size_title=fig_dict['em_title_font_size'])

        if ((6550 in em_line_fit_dict['ln_list']) & (6565 in em_line_fit_dict['ln_list']) &
                (6585 in em_line_fit_dict['ln_list'])):
            ax_ha_nii_spec = fig.add_axes([
                fig_dict['ha_nii_left_align'],
                fig_dict['ha_nii_bottom_align'],
                fig_dict['ha_nii_width'],
                fig_dict['ha_nii_height'],
            ])
            ax_ha_nii_res_spec = fig.add_axes([
                fig_dict['ha_nii_res_left_align'],
                fig_dict['ha_nii_res_bottom_align'],
                fig_dict['ha_nii_res_width'],
                fig_dict['ha_nii_res_height'],
            ])
            plotting_tools.SpecPlotTools.plot_em_line_spec(ax=ax_ha_nii_spec, em_line_fit_dict=em_line_fit_dict,
                                                           line_list=[6550, 6565, 6585],
                                                           left_offset=fig_dict['ha_nii_left_lim_offst'],
                                                           right_offset=fig_dict['ha_nii_right_lim_offst'],
                                                           display_legend=False,
                                                           display_x_label=True,
                                                           ax_res=ax_ha_nii_res_spec,
                                                           font_size_label=fig_dict['em_label_size'],
                                                           font_size_title=fig_dict['em_title_font_size'])

        if 6302 in em_line_fit_dict['ln_list']:
            ax_oi6302_spec = fig.add_axes([
                fig_dict['oi6302_left_align'],
                fig_dict['oi6302_bottom_align'],
                fig_dict['oi6302_width'],
                fig_dict['oi6302_height'],
            ])
            ax_oi6302_res_spec = fig.add_axes([
                fig_dict['oi6302_res_left_align'],
                fig_dict['oi6302_res_bottom_align'],
                fig_dict['oi6302_res_width'],
                fig_dict['oi6302_res_height'],
            ])
            plotting_tools.SpecPlotTools.plot_em_line_spec(ax=ax_oi6302_spec, em_line_fit_dict=em_line_fit_dict,
                                                           line_list=[6302],
                                                           left_offset=fig_dict['oi6302_left_lim_offst'],
                                                           right_offset=fig_dict['oi6302_right_lim_offst'],
                                                           display_legend=False,
                                                           display_x_label=False,
                                                           display_y_label=False,
                                                           ax_res=ax_oi6302_res_spec,
                                                           font_size_label=fig_dict['em_label_size'],
                                                           font_size_title=fig_dict['em_title_font_size'])

        if (6718 in em_line_fit_dict['ln_list']) & (6733 in em_line_fit_dict['ln_list']):
            ax_sii_spec = fig.add_axes([
                fig_dict['sii_left_align'],
                fig_dict['sii_bottom_align'],
                fig_dict['sii_width'],
                fig_dict['sii_height'],
            ])
            ax_sii_res_spec = fig.add_axes([
                fig_dict['sii_res_left_align'],
                fig_dict['sii_res_bottom_align'],
                fig_dict['sii_res_width'],
                fig_dict['sii_res_height'],
            ])
            plotting_tools.SpecPlotTools.plot_em_line_spec(ax=ax_sii_spec, em_line_fit_dict=em_line_fit_dict,
                                                           line_list=[6718, 6733],
                                                           left_offset=fig_dict['sii_left_lim_offst'],
                                                           right_offset=fig_dict['sii_right_lim_offst'],
                                                           display_legend=False,
                                                           display_x_label=True,
                                                           display_y_label=False,
                                                           ax_res=ax_sii_res_spec,
                                                           font_size_label=fig_dict['em_label_size'],
                                                           font_size_title=fig_dict['em_title_font_size'])

        # ax_red_bump = fig.add_axes([
        #     fig_dict['red_bump_left_align'],
        #     fig_dict['red_bump_bottom_align'],
        #     fig_dict['red_bump_width'],
        #     fig_dict['red_bump_height'],
        # ])
        #
        # ax_hei6680_spec = fig.add_axes([
        #     fig_dict['hei6680_left_align'],
        #     fig_dict['hei6680_bottom_align'],
        #     fig_dict['hei6680_width'],
        #     fig_dict['hei6680_height'],
        # ])
        # plotting_tools.SpecPlotTools.plot_red_bump(ax=ax_red_bump, ppxf_fit_dict=ppxf_fit_dict,
        #                                            font_size_label=fig_dict['em_label_size'],
        #                                            font_size_title=fig_dict['em_title_font_size'])
        #
        # plotting_tools.SpecPlotTools.plot_stellar_hei6680(ax=ax_hei6680_spec, ppxf_fit_dict=ppxf_fit_dict,
        #                                                   y_axis_scale=1e16, font_size_label=fig_dict['em_label_size'],
        #                                                   display_label=True, font_size_title=fig_dict['em_title_font_size'],
        #                                                   display_y_label=False, display_x_label=True)

    def plot_kcwi_muse_spec(self, fig, fig_dict, kcwi_spec_dict=None, muse_spec_dict=None):

        # add sed axis
        ax_spec = fig.add_axes([
            fig_dict['spec_left_align'],
            fig_dict['spec_bottom_align'],
            fig_dict['spec_width'],
            fig_dict['spec_height'],
        ])

        if kcwi_spec_dict is not None:
            kcwi_flux = kcwi_spec_dict['native_spec_flx'].to((1e-20) * u.erg / u.s / u.cm / u.cm / u.Angstrom)
            ax_spec.plot(kcwi_spec_dict['native_wave'], kcwi_flux, linewidth=2)

        if muse_spec_dict is not None:
            muse_flux = muse_spec_dict['native_spec_flx'].to((1e-20) * u.erg / u.s / u.cm / u.cm / u.Angstrom)
            ax_spec.plot(muse_spec_dict['native_wave'], muse_flux, linewidth=2)

        ax_spec.set_xscale('log')
        ax_spec.set_yscale('log')

        if kcwi_spec_dict is not None:
            ax_blue_bump = fig.add_axes([
                fig_dict['blue_bump_left_align'],
                fig_dict['blue_bump_bottom_align'],
                fig_dict['blue_bump_width'],
                fig_dict['blue_bump_height'],
            ])

            plotting_tools.SpecPlotTools.plot_spec_feature_window(
                ax=ax_blue_bump, spec_dict=kcwi_spec_dict, font_size_label=fig_dict['spec_title_font_size'],
                spec_feature='Blue Bump')

        if muse_spec_dict is not None:

            ax_red_bump = fig.add_axes([
                fig_dict['red_bump_left_align'],
                fig_dict['red_bump_bottom_align'],
                fig_dict['red_bump_width'],
                fig_dict['red_bump_height'],
            ])



            plotting_tools.SpecPlotTools.plot_spec_feature_window(
                ax=ax_red_bump, spec_dict=muse_spec_dict, font_size_label=fig_dict['spec_title_font_size'],
                spec_feature='Red Bump')



            ax_ha = fig.add_axes([
                fig_dict['ha_left_align'],
                fig_dict['ha_bottom_align'],
                fig_dict['ha_width'],
                fig_dict['ha_height'],
            ])

            plotting_tools.SpecPlotTools.plot_spec_feature_window(
                ax=ax_ha, spec_dict=muse_spec_dict, font_size_label=fig_dict['spec_title_font_size'],
                spec_feature='Halpha')





    @staticmethod
    def phangs_holistic_viewer1(ra, dec, target_name=None, phot_visual_access=None,
                                plot_rad_profile=False, plot_sed=False,
                                plot_muse=True, ppxf_fit_dict=None):
        """

        This method creates a holistic inspection plot for one coordinate.

        This is based on the phangs data access tools and therefore not universal for any objects.

        """

        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.holistic_viewer1_param_dic)

        if phot_visual_access is None:
            phot_visual_access = PhotVisualizer(target_name=target_name)

        # create the overview plot
        phot_visual_access.plot_hst_overview_panel(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic,
                                                   ra_box=ra, dec_box=dec)

        # plot environment zoom in panels
        phot_visual_access.plot_zoom_in_panel_group(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic,
                                                    ra=ra, dec=dec)

        # plot postage stamps
        phot_visual_access.plot_img_stamps(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic, ra=ra, dec=dec,
                                           plot_rad_profile=plot_rad_profile)

        # plot sed estimation
        if plot_sed:
            phot_visual_access.plot_sed_panel(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic, ra=ra, dec=dec)

        # # get_EW estimation
        # phot_visual_access.compute_ha_ew(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic, ra=ra, dec=dec)

        # # get MUSE spectrum from region
        if plot_muse:
            if ppxf_fit_dict is None:
                rad_arcsec = phangs_info.muse_obs_res_dict[phot_visual_access.spec_target_name]['copt_res'] / 2
                spec_dict = phot_visual_access.extract_muse_spec_circ_app(ra=ra, dec=dec,
                                                                          rad_arcsec=rad_arcsec, wave_range=None,
                                                                          res='copt')
                ppxf_fit_dict = spec_tools.SpecTools.fit_ppxf2spec(spec_dict=spec_dict,
                                                                   target=phot_visual_access.spec_target_name,
                                                                   sps_name='fsps', age_range=None, metal_range=None)
            phot_visual_access.plot_muse_spec(fig=fig, fig_dict=plot_params.holistic_viewer1_param_dic, ra=ra, dec=dec,
                                              ppxf_fit_dict=ppxf_fit_dict)

        return fig

    def phangs_holistic_viewer2(self, ra, dec, plot_rad_profile=False, plot_sed=False):
        """

        This method creates a holistic inspection plot for one coordinate.

        This is based on the phangs data access tools and therefore not universal for any objects.

        """

        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.holistic_viewer2_param_dic)

        # create the overview plot
        self.plot_hst_overview_panel(fig=fig, fig_dict=plot_params.holistic_viewer2_param_dic,
                                                   ra_box=ra, dec_box=dec)

        # plot environment zoom in panels
        self.plot_zoom_in_panel_group_extra_nircam(fig=fig, fig_dict=plot_params.holistic_viewer2_param_dic,
                                                    ra=ra, dec=dec)

        # plot postage stamps
        self.plot_img_stamps_all(fig=fig, fig_dict=plot_params.holistic_viewer2_param_dic, ra=ra, dec=dec,
                                           plot_rad_profile=plot_rad_profile,
                                               individual_band_list=plot_params.holistic_viewer2_param_dic['individual_band_list'])

        # plot sed estimation
        if plot_sed:
            self.plot_sed_panel(fig=fig, fig_dict=plot_params.holistic_viewer2_param_dic, ra=ra, dec=dec,
                                              individual_band_list=plot_params.holistic_viewer2_param_dic['individual_band_list'])

        return fig

    def phangs_holistic_viewer3(self, ra, dec, plot_rad_profile=False, plot_sed=False, radio_ellipse_dict=None):
        """

        This method creates a holistic inspection plot for one coordinate.

        This is based on the phangs data access tools and therefore not universal for any objects.

        """

        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.holistic_viewer3_param_dic)

        # create the overview plot
        self.plot_hst_overview_panel(fig=fig, fig_dict=plot_params.holistic_viewer3_param_dic,
                                                   ra_box=ra, dec_box=dec)

        # plot environment zoom in panels
        self.plot_zoom_in_panel_group_extra_nircam(fig=fig, fig_dict=plot_params.holistic_viewer3_param_dic,
                                                   ra=ra, dec=dec)

        # add xray
        self.plot_x_ray_zoom_in(fig=fig, fig_dict=plot_params.holistic_viewer3_param_dic, ra=ra, dec=dec)

        # add radio
        self.plot_radio_cont_zoom_in(fig=fig, fig_dict=plot_params.holistic_viewer3_param_dic, ra=ra, dec=dec, radio_ellipse_dict=radio_ellipse_dict)


        # plot postage stamps
        self.plot_img_stamps_all(fig=fig, fig_dict=plot_params.holistic_viewer3_param_dic, ra=ra, dec=dec,
                                           plot_rad_profile=plot_rad_profile,
                                               individual_band_list=plot_params.holistic_viewer3_param_dic['individual_band_list'])

        # plot sed estimation
        if plot_sed:
            self.plot_sed_panel(fig=fig, fig_dict=plot_params.holistic_viewer3_param_dic, ra=ra, dec=dec,
                                              individual_band_list=plot_params.holistic_viewer3_param_dic['individual_band_list'])

        return fig

    def wr_inspector(self, ra, dec, plot_sed=False, plot_x_ray=True, kcwi_spec_dict=None, muse_spec_dict=None):
        """

        This method creates an inspection plot to identify WR features

        """

        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.wr_inspector_param_dic)

        # create the overview plot
        self.plot_hst_overview_panel(fig=fig, fig_dict=plot_params.wr_inspector_param_dic,
                                                   ra_box=ra, dec_box=dec)

        # plot environment zoom in panels
        self.plot_zoom_in_panel_group_extra_nircam(fig=fig, fig_dict=plot_params.wr_inspector_param_dic,
                                                   obs_list=['hst_broad_band', 'hst_ha', 'nircam1', 'miri1'],
                                                   ra=ra, dec=dec)

        # add xray
        if plot_x_ray:
            self.plot_x_ray_zoom_in(fig=fig, fig_dict=plot_params.wr_inspector_param_dic, ra=ra, dec=dec)



        self.plot_kcwi_muse_spec(fig=fig, fig_dict=plot_params.wr_inspector_param_dic,
                                 muse_spec_dict=muse_spec_dict, kcwi_spec_dict=kcwi_spec_dict)

        # plot sed estimation
        if plot_sed:
            self.plot_sed_panel(fig=fig, fig_dict=plot_params.wr_inspector_param_dic, ra=ra, dec=dec,
                                              individual_band_list=plot_params.wr_inspector_param_dic['individual_band_list'])

        return fig



    def vms_quick_inspector(self, ra, dec, kcwi_spec_dict=None, muse_spec_dict=None):
        """

        This method creates an inspection plot to identify WR features

        """

        # create figure
        fig = plotting_tools.AxisTools.init_fig(fig_dict=plot_params.vms_quick_inspector_param_dic)

        # create the overview plot
        self.plot_hst_overview_panel(fig=fig, fig_dict=plot_params.vms_quick_inspector_param_dic,
                                                   ra_box=ra, dec_box=dec)

        # plot environment zoom in panels
        self.plot_zoom_in_panel_group_extra_nircam(fig=fig, fig_dict=plot_params.vms_quick_inspector_param_dic,
                                                   obs_list=['hst_ha', 'nircam1'],
                                                   ra=ra, dec=dec)

        self.plot_kcwi_muse_spec(fig=fig, fig_dict=plot_params.vms_quick_inspector_param_dic,
                                 muse_spec_dict=muse_spec_dict, kcwi_spec_dict=kcwi_spec_dict)

        return fig





