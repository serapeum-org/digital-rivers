"""Terrain analysis module.

This module provides the ``Terrain`` class for raster-based terrain
visualisation and analysis: color relief, hill shade, slope, and aspect.
All heavy lifting is delegated to GDAL's ``DEMProcessing`` utility.
"""
import os
import numpy as np
from pandas import DataFrame
from typing import List, Union
import tempfile
import uuid
from osgeo import gdal
from pyramids.dataset import Dataset

CREATION_OPTIONS = ["COMPRESS=DEFLATE", "PREDICTOR=2"]


class Terrain(Dataset):
    """Terrain analysis tools built on GDAL ``DEMProcessing``.

    Wraps a single- or multi-band raster and exposes convenience methods
    for color relief, hill shade, slope, and aspect computation.

    Args:
        raster: File path or GDAL dataset to open.
        access: ``"read"`` (default) or ``"write"``.
    """

    def __init__(self, raster: Union[str, gdal.Dataset], access: str = "read"):
        super().__init__(raster, access)

    def color_relief(
        self, band: int = 0, path: str = None, color_table: DataFrame = None, **kwargs
    ) -> "Dataset":
        """Create a color relief for a band in the Dataset.

        A color relief raster is a raster image where each pixel's value is mapped to a specific color based on a
        predefined color palette or color table.

        Args:
            band: int, default is 0.
                band index.
            path: str, default is None.
                path to save the color relief raster.
            color_table: DataFrame, default is None.
                DataFrame with columns: band, values, color
                    ```text
                      values    color
                    0      1  #709959
                    1      2  #F2EEA2
                    2      3  #F2CE85
                    3      1  #C28C7C
                    4      2  #D6C19C
                    5      3  #D6C19C
                    ```
                or DataFrame with columns: values, red, green, blue, alpha, (the alpha column is optional)
                    ```text
                      values    red  green   blue  alpha
                    0      1    112    153     89    255
                    1      2    242    238    162    255
                    2      3    242    206    133    255
                    3      1    194    140    124    255
                    4      2    214    193    156    255
                    5      3    214    193    156    255
                    ```
        Returns:
            Dataset:
                Dataset with the color relief with four bands read, green, blue, and alpha.

        Examples:
            - First create a one band dataset, consisting of 10 columns and 10 rows, with random values between 0 and 15.
                ```python
                >>> import numpy as np
                >>> arr = np.random.randint(0, 15, size=(10, 10))
                >>> dataset = Dataset.create_from_array(arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326)
                ```
            - Now let's create the color table using hex colors.
                ```python
                >>> color_hex = ["#709959", "#F2EEA2", "#F2CE85", "#C28C7C", "#D6C19C"]
                >>> values = [1, 3, 5, 7, 9]
                >>> df = pd.DataFrame(columns=["values", "color"])
                >>> df.loc[:, "values"] = values
                >>> df.loc[:, "color"] = color_hex
                ```
            - Now let's create the color relief for the dataset using the color table `DataFrame`.
                ```python
                >>> color_relief = dataset.color_relief(band=0, color_table=df)
                >>> print(color_relief) # doctest: +SKIP
                <BLANKLINE>
                            Cell size: 0.05
                            Dimension: 10 * 10
                            EPSG: 4326
                            Number of Bands: 4
                            Band names: ['Band_1', 'Band_2', 'Band_3', 'Band_4']
                            Mask: None
                            Data type: byte
                            projection: GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]
                            Metadata: {}
                            File: ...
                <BLANKLINE>
                >>> print(color_relief.band_color)
                {0: 'red', 1: 'green', 2: 'blue', 3: 'alpha'}
                ```
            - The result color relief dataset will have 4 bands red, green, blue, and alpha. with values from 0 to 255.
            - To plot the color relief dataset, you can use the `plot` method. but you need to provide the the rgb indices
                with the alpha index as the fourth index, otherwise the alpha band will be missing.
                ```python
                >>> fig, ax = color_relief.plot(rgb=[0, 1, 2, 3])

                ```
            ![color-relief](./../_images/dataset/color-relief.png)

        See Also:
            Dataset.hill_shade: create a hill-shade for a band in the Dataset.
        """
        if path is None:
            driver = "MEM"
            path = ""
        else:
            driver = "GTiff"
        color_df = self._process_color_table(color_table)

        temp_dir = tempfile.mkdtemp()
        color_table_path = os.path.join(temp_dir, f"{uuid.uuid1()}.txt")
        color_df.to_csv(color_table_path, index=False, header=False)

        options = gdal.DEMProcessingOptions(
            band=band + 1,
            format=driver,
            colorFilename=color_table_path,
            addAlpha=True,
            creationOptions=["COMPRESS={}".format("DEFLATE"), "PREDICTOR={}".format(2)],
            **kwargs,
        )
        dst = gdal.DEMProcessing(path, self.raster, "color-relief", options=options)
        color_relief = Dataset(dst, access="write")
        color_relief.band_color = {0: "red", 1: "green", 2: "blue", 3: "alpha"}
        return color_relief

    def hill_shade(
        self,
        band: int = 0,
        azimuth: Union[int, float, List[int]] = 315,
        altitude: Union[int, float, List[int]] = 45,
        vertical_exaggeration: Union[int, float, List[int]] = 1,
        scale: Union[int, float, List[int]] = 1,
        path: str = None,
        weights: List[int] = None,
        **kwargs,
    ) -> "Dataset":
        """Create hill-shade.

        Hillshade is a technique used in digital elevation modeling (DEM) to create a grayscale representation of a
        terrain's surface that simulates the effect of sunlight falling across the landscape.
        This technique helps to visualize the shape and features of the terrain by highlighting the variations in
        elevation and the slope of the surface.

        Hillshade calculates the illumination of each pixel based on the slope (gradient) and aspect (direction) of the
        terrain surface relative to a specified light source.

        The main parameters influencing the hillshade effect are:
        - Light source direction (Azimuth): the azimuth angle of the light source, which is the angle between the light
            source
        - Light source elevation (altitude): the source of light elevation, it is measured in degrees from the horizon.
        - Vertical exaggeration (Z-factor): the vertical exaggeration is used to emphasize the vertical features of the
            terrain.

        Notes:
            if the `hill_shade` parameters are given as lists then the hill shade will be calculated for each set
            of parameter and then the average will be returned.

        Args:
            band: int
                band index.
            azimuth: Union[List, int, float]
                The source of light direction, it is measured clockwise from the north. zero means from north to south.
                45 degrees means from the northeast to the southwest.
            altitude: Union[List, int, float]
                The source of light elevation, it is measured in degrees from the horizon. zero means from the horizon.
                90 degrees means from the zenith.
                the overall image gets brighter as the light source gets closer to the zenith. The brightest slopes/DEM
                features will be perpendicular to the light source, and the darkest will be angled 90˚ or more away.
            vertical_exaggeration: Union[List, int, float]
                Vertical exaggeration, the vertical exaggeration It is used to emphasize the
                vertical features of the terrain.
            scale: Union[List, int, float]
                the scale is the ratio of vertical units to horizontal. If the horizontal unit of the source DEM is
                degrees (e.g Lat/Long WGS84 projection), you can use scale=111120 if the vertical units are meters
                (or scale=370400 if they are in feet).
            path: str, optional, default is None
                path to save the hill-shade raster.
            weights: List[int], default is None.
                list of weights to combine the hill-shades if the other parameters are given as lists, an average hill
                shade will be calculated based on the weights. if None, the weights will be equal.
            **kwargs:
                multi_directional: bool
                    if True, the hill shade will be calculated for multiple azimuth values [225, 270, 315, 360] each with a
                    altitude of 30 degrees, and then the average will be returned. with multi_directional = True any given
                    azimuth will be ignored.
                    For more details visit: https://pubs.usgs.gov/of/1992/of92-422/of92-422.pdf
                combined: bool
                    combined shading, a combination of slope and oblique shading.
                igor: bool
                    shading which tries to minimize effects on other map features beneath. with `igor=True` the altitude
                    will be calculated ignored.
                    For more details visit: https://maperitive.net/docs/Commands/GenerateReliefImageIgor.html

        Returns:
            Dataset: 8-bit
                Dataset with the hill-shade created.

        Examples:
            - First create a one band dataset, consisting of 10 columns and 10 rows, with random values between 0 and 15.
                ```python
                >>> import numpy as np
                >>> arr = np.random.randint(0, 15, size=(100, 100))
                >>> dataset = Dataset.create_from_array(arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326)

                >>> hill_shade = dataset.hill_shade(
                ...     band=0, altitude=45, azimuth=315, vertical_exaggeration=1, scale=1
                ... )

                >>> print(hill_shade.dtype) # doctest: +SKIP
                ['byte']
                >>> hill_shade.plot() # doctest: +SKIP
                ```
                ![hill-shade](./../_images/dataset/hill-shade.png)
                ```python
                >>> hill_shade.stats() # doctest: +SKIP
                        min    max       mean        std
                Band_1  1.0  223.0  58.880951  71.079056
                ```
            - You can also provide the function with a list os values for each parameter, then the functions will
                calculate the hill shade for each set of parameters and then the average will be returned.
                ```python
                >>> hill_shade = dataset.hill_shade(
                ...     band=0, azimuth=[315, 45], altitude=[45, 45], vertical_exaggeration=[1, 1], scale=[1, 1]
                ... )

                >>> hill_shade.plot() # doctest: +SKIP

                ```
                ![hill-shade-multi](./../_images/dataset/hill-shade-multi.png)

        See Also:
            Dataset.color_relief: create a color relief for a band in the Dataset.
            Dataset.slope: create a slope for a band in the Dataset.
        """
        if "multi_directional" in kwargs:
            if not isinstance(kwargs["multi_directional"], bool):
                raise ValueError("The multi_directional parameter must be a boolean.")
            if kwargs["multi_directional"]:
                multi_directional = True
                azimuth = None
                # altitude, vertical_exaggeration, scale = None, None, None,
            else:
                multi_directional = False

            kwargs.pop("multi_directional")
            kwargs["multiDirectional"] = multi_directional
        if "igor" in kwargs:
            if not isinstance(kwargs["igor"], bool):
                raise ValueError("The igor parameter must be a boolean.")
            if kwargs["igor"]:
                altitude = None

        # if not (
        #     type(azimuth)
        #     is type(altitude)
        #     is type(vertical_exaggeration)
        #     is type(scale)
        # ):
        #     raise ValueError(
        #         f"The azimuth, altitude, vertical_exaggeration, and scale parameter must be of the same type. Given"
        #         f" azimuth: {type(azimuth)}, altitude: {type(altitude)}, vertical_exaggeration: {type(vertical_exaggeration)},"
        #         f"scale: {type(scale)}"
        #     )

        if path is None:
            driver = "MEM"
            path = ""
        else:
            driver = "GTiff"

        # if parameters are lists
        if isinstance(azimuth, list):
            if (
                    len(azimuth)
                    != len(altitude)
                    != len(vertical_exaggeration)
                    != len(scale)
            ):
                raise ValueError(
                    "The length of the light source angle and elevation must be the same."
                )
        else:
            azimuth = [azimuth]
            altitude = [altitude]
            vertical_exaggeration = [vertical_exaggeration]
            scale = [scale]

        # get the hill shade for all the parameters
        hill_shades: List[gdal.Dataset] = []
        for az, alt, ver_ex, scale_1 in zip(
                azimuth, altitude, vertical_exaggeration, scale
        ):
            dst = self._create_hill_shade(
                band, driver, az, alt, ver_ex, scale_1, path, **kwargs
            )
            hill_shades.append(dst)

        if len(hill_shades) > 1:
            if weights is None:
                weights = np.ones(len(azimuth))
            weights = np.array(weights) / np.sum(weights)
            hill_shades_arr: List[np.ndarray] = [
                hill_shade.ReadAsArray() for hill_shade in hill_shades
            ]
            combined_hillshade = np.average(hill_shades_arr, axis=0, weights=weights)
            combined_hillshade = np.clip(combined_hillshade, 0, 255).astype(np.uint8)
            hill_shade = Dataset.dataset_like(self, combined_hillshade)
        else:
            hill_shade = Dataset(hill_shades[0], access="write")

        hill_shade.band_color = {0: "gray_index"}

        return hill_shade

    def _create_hill_shade(
        self,
        band: int,
        driver: str,
        azimuth: Union[int, float] = 315,
        altitude: Union[int, float] = 45,
        vertical_exaggeration: Union[int, float] = 1,
        scale: Union[int, float] = 1,
        path: str = None,
        **kwargs,
    ) -> gdal.Dataset:
        """Run a single GDAL ``DEMProcessing("hillshade")`` call.

        Args:
            band: Zero-based band index.
            driver: GDAL driver name (``"MEM"`` or ``"GTiff"``).
            azimuth: Light-source azimuth in degrees clockwise from
                north.
            altitude: Light-source elevation in degrees above horizon.
            vertical_exaggeration: Z-factor for vertical emphasis.
            scale: Ratio of vertical to horizontal units.
            path: Output file path (empty string for in-memory).
            **kwargs: Forwarded to ``gdal.DEMProcessingOptions``.

        Returns:
            gdal.Dataset: Raw GDAL dataset with the computed hill shade.
        """
        options = gdal.DEMProcessingOptions(
            band=band + 1,
            format=driver,
            azimuth=azimuth,
            altitude=altitude,
            zFactor=vertical_exaggeration,
            scale=scale,
            creationOptions=["COMPRESS=LZW"],
            **kwargs,
        )
        dst = gdal.DEMProcessing(path, self.raster, "hillshade", options=options)

        return dst

    def slope(
        self,
        band: int = 0,
        scale: Union[int, float, List[int]] = 1,
        slope_format: str = "degree",
        path: str = None,
        algorithm: str = None,
        creation_options: List[str] = None,
        **kwargs,
    ) -> "Dataset":
        """Compute the slope of the terrain surface.

        Uses GDAL ``DEMProcessing`` to calculate the slope (rate of
        elevation change) for every cell.

        Args:
            band: Zero-based band index. Defaults to 0.
            scale: Ratio of vertical to horizontal units.  Use
                ``111120`` when the horizontal CRS is in degrees and
                vertical units are metres.  Defaults to 1.
            slope_format: Output format — ``"degree"`` (default) or
                ``"percent"``.
            algorithm: Slope algorithm.  One of ``"Horn"``,
                ``"ZevenbergenThorne"``, or ``None`` (GDAL default).
                Zevenbergen-Thorne suits smooth landscapes; Horn
                performs better on rough terrain.
            path: If given, write the result to this GeoTIFF path.
                Otherwise the raster is created in memory.
            creation_options: GDAL creation options.  Defaults to
                ``['COMPRESS=DEFLATE', 'PREDICTOR=2']``.
            **kwargs: Forwarded to ``gdal.DEMProcessingOptions``.

        Returns:
            Dataset: Single-band ``float32`` raster with slope values.
                No-data value is ``-9999.0``.

        Examples:
            - First create a one band dataset, consisting of 10 columns
                and 10 rows, with random values between 0 and 15.
                ```python
                >>> import numpy as np
                >>> arr = np.random.randint(0, 15, size=(10, 10))
                >>> dataset = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
                ... )
                ```
            - Now let's create the slope for the dataset.
                ```python
                >>> slope = dataset.slope()
                >>> fig, ax = slope.plot()

                ```
                ![slope](./../_images/dataset/slope.png)

        See Also:
            Terrain.hill_shade: Create a hill-shade for a band in the
                Dataset.
            Terrain.color_relief: Create a color relief for a band in
                the Dataset.
        """
        if path is None:
            driver = "MEM"
            path = ""
        else:
            driver = "GTiff"

        if creation_options is None:
            creation_options = CREATION_OPTIONS.copy()

        options = gdal.DEMProcessingOptions(
            band=band + 1,
            format=driver,
            alg=algorithm,
            slopeFormat=slope_format,
            scale=scale,
            creationOptions=creation_options,
            **kwargs,
        )
        dst = gdal.DEMProcessing(path, self.raster, "slope", options=options)
        src = Dataset(dst, access="write")

        return src

    def aspect(
        self,
        band: int = 0,
        scale: Union[int, float, List[int]] = 1,
        vertical_exaggeration: Union[int, float, List[int]] = 1,
        zero_flat_surface: bool = False,
        algorithm: str = None,
        path: str = None,
        creation_options: List[str] = None,
        **kwargs,
    ) -> "Dataset":
        """Compute the aspect (slope direction) of the terrain surface.

        Uses GDAL ``DEMProcessing`` to calculate the compass direction
        of the steepest downhill slope for every cell.  Values range
        from 0° (north) clockwise to 360°.

        Args:
            band: Zero-based band index. Defaults to 0.
            scale: Ratio of vertical to horizontal units.  Use
                ``111120`` when the horizontal CRS is in degrees and
                vertical units are metres.  Defaults to 1.
            vertical_exaggeration: Z-factor used to emphasise vertical
                features.  Defaults to 1.
            zero_flat_surface: If ``True`` flat areas get an aspect of
                0°.  If ``False`` (default) flat areas receive the
                no-data value.
            algorithm: Aspect algorithm.  One of ``"Horn"``,
                ``"ZevenbergenThorne"``, or ``None`` (GDAL default).
            path: If given, write the result to this GeoTIFF path.
                Otherwise the raster is created in memory.
            creation_options: GDAL creation options.  Defaults to
                ``['COMPRESS=DEFLATE', 'PREDICTOR=2']``.
            **kwargs: Forwarded to ``gdal.DEMProcessingOptions``.

        Returns:
            Dataset: Single-band ``float32`` raster with aspect values
                in degrees (0–360).  No-data value is ``-9999.0``.

        Examples:
            - Create a small raster and compute its aspect.
                ```python
                >>> import numpy as np
                >>> arr = np.random.randint(0, 15, size=(10, 10))
                >>> dataset = Dataset.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326
                ... )
                ```
            - Compute the aspect raster.
                ```python
                >>> aspect = dataset.aspect()
                >>> fig, ax = aspect.plot()

                ```
                ![aspect](./../_images/dataset/aspect.png)

        See Also:
            Terrain.hill_shade: Create a hill-shade for a band in the
                Dataset.
            Terrain.slope: Compute the slope of the terrain surface.
        """
        if path is None:
            driver = "MEM"
            path = ""
        else:
            driver = "GTiff"

        if creation_options is None:
            creation_options = CREATION_OPTIONS.copy()

        options = gdal.DEMProcessingOptions(
            band=band + 1,
            format=driver,
            alg=algorithm,
            scale=scale,
            zFactor=vertical_exaggeration,
            zeroForFlat=zero_flat_surface,
            creationOptions=creation_options,
            **kwargs,
        )
        dst = gdal.DEMProcessing(path, self.raster, "aspect", options=options)
        src = Dataset(dst, access="write")

        return src