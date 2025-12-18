"""
This is the frontend of the Sternenfabrik package
"""
from sternenfabrik.plot_fabrik import PlotFabrik


class StarFactory(PlotFabrik):
    """
    This is the frontend. In order to be consistent with future versions it is basically the child of all parent
    calsses that harbor functionalities.
    """

    def __init__(self, target_name=None, phot_hst_target_name=None, phot_hst_ha_cont_sub_target_name=None,
                 phot_nircam_target_name=None, phot_miri_target_name=None, phot_astrosat_target_name=None,
                 x_target_name=None, radio_target_name=None,
                 nircam_data_ver='v1p1p1', miri_data_ver='v1p1p1', astrosat_data_ver='v1p0'):
        PlotFabrik.__init__(
            self, target_name=target_name, phot_hst_target_name=phot_hst_target_name,
            phot_hst_ha_cont_sub_target_name=phot_hst_ha_cont_sub_target_name,
            phot_nircam_target_name=phot_nircam_target_name, phot_miri_target_name=phot_miri_target_name,
            phot_astrosat_target_name=phot_astrosat_target_name, x_target_name=x_target_name, radio_target_name=radio_target_name,
            nircam_data_ver=nircam_data_ver, miri_data_ver=miri_data_ver, astrosat_data_ver=astrosat_data_ver)
