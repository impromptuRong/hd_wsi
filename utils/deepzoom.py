import math
from io import BytesIO
from PIL import Image


def get_dzi(image_size, tile_size=254, overlap=1, format='jpeg'):
    """ Return a string containing the XML metadata for the .dzi file.
        image_size: (w, h)
        tile_size: tile size
        overlap: overlap size
        format: the format of the individual tiles ('png' or 'jpeg')
    """
    import xml.etree.ElementTree as ET
    image = ET.Element(
        'Image',
        TileSize=str(tile_size),
        Overlap=str(overlap),
        Format=format,
        xmlns='http://schemas.microsoft.com/deepzoom/2008',
    )
    w, h = image_size
    ET.SubElement(image, 'Size', Width=str(w), Height=str(h))
    tree = ET.ElementTree(element=image)
    buf = BytesIO()
    tree.write(buf, encoding='UTF-8')

    return buf.getvalue().decode('UTF-8')


class DeepZoomGenerator:
    """Generates Deep Zoom tiles and metadata."""

    def __init__(self, osr, tile_size=254, overlap=1, limit_bounds=False, image_size=None, format='jpeg'):
        """ Create a DeepZoomGenerator wrapping an OpenSlide object.
        osr:          a Slide object.
        tile_size:    the width and height of a single tile.  For best viewer
                      performance, tile_size + 2 * overlap should be a power
                      of two.
        overlap:      the number of extra pixels to add to each interior edge
                      of a tile.
        limit_bounds: True to render only the non-empty slide region.
                      (We don't use limit_bounds)
        image_size:   give a arbitary image size, default use w, h from page0.
        """
        self._osr = osr
        self.tile_size = tile_size
        self.overlap = overlap
        if limit_bounds:
            # print(f"We ignore limit_bounds and display everything!")
            limit_bounds = False
        self._limit_bounds = limit_bounds
        self._bg_color = '#ffffff'  # Slide background color
        self.default_format = self.format = format

        # Deep Zoom level
        # z_size = image_size or osr.level_dimensions[0]
        # z_dimensions = [z_size]
        # while z_size[0] > 1 or z_size[1] > 1:
        #     z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)
        #     z_dimensions.append(z_size)
        # self.dz_dimensions = tuple(reversed(z_dimensions))
        self.dz_dimensions = osr.deepzoom_dims(image_size)
        self._dz_levels = len(self.dz_dimensions)  # Deep Zoom level count

        self.page_tile_indices = []
        for dz_level, (wz, hz) in enumerate(self.dz_dimensions):
            d = 2 ** (self._dz_levels - dz_level - 1)
            page = osr.get_resize_level(d, downsample_only=True)
            factor = d / osr.level_downsamples[page]
            tile_size = round(self.tile_size * factor)
            overlap = round(self.overlap * factor)

            coords_dzi = osr.deepzoom_coords(self.tile_size, self.overlap, image_size=(wz, hz))
            coords_osr = osr.deepzoom_coords(tile_size, overlap, image_size=page)
            # coords_osr = (coords_dzi * factor).round().clip()

            self.page_tile_indices.append([page, coords_osr, coords_dzi])

    def __repr__(self):
        return '{}({!r}, tile_size={!r}, overlap={!r}, limit_bounds={!r})'.format(
            self.__class__.__name__,
            self._osr,
            self.tile_size,
            self.overlap,
            self._limit_bounds,
        )

    @property
    def level_count(self):
        """The number of Deep Zoom levels in the image."""
        return self._dz_levels

    @property
    def level_tiles(self):
        """A list of (tiles_x, tiles_y) tuples for each Deep Zoom level."""
        # Tile
        def tiles(z_lim):
            return int(math.ceil(z_lim / self.tile_size))

        return tuple(
            (tiles(z_w), tiles(z_h)) for z_w, z_h in self.dz_dimensions
        )

    @property
    def level_dimensions(self):
        """A list of (pixels_x, pixels_y) tuples for each Deep Zoom level."""
        return self.dz_dimensions

    @property
    def tile_count(self):
        """The total number of Deep Zoom tiles in the image."""
        return sum(t_cols * t_rows for t_cols, t_rows in self.level_tiles)

    def get_tile(self, level, address, format=None):
        """Return an RGB(A) PIL.Image for a tile.
        level:     the Deep Zoom level.
        address:   the address of the tile within the level as a (col, row)
                   tuple.
        """
        format = format or self.default_format
        page, coord, z_size = self.get_tile_info(level, address)
        tile = self._osr.get_patch(x=coord, level=page)
#         x0, y0, w, h = coord
#         scale = self._osr.level_downsamples[page]
#         args = (int(x0 * scale), int(y0 * scale)), page, (int(w), int(h))
#         tile = self._osr.fh.read_region(*args)

        # Apply on solid background if it's a rgba image
        if tile.mode == 'RGBA' and format == 'jpeg':
            bg = Image.new('RGB', tile.size, self._bg_color)
            tile = Image.composite(tile, bg, tile)

        # Scale to the correct size
        if tile.size != z_size:
            # Image.Resampling added in Pillow 9.1.0
            # Image.LANCZOS removed in Pillow 10
            tile.thumbnail(z_size, getattr(Image, 'Resampling', Image).LANCZOS)

        return tile

    def get_tile_info(self, level, address):
        # Check parameters
        if level < 0 or level >= self._dz_levels:
            raise ValueError(f"Invalid level {level}")
        col, row = address
        
        # Get preferred slide page
        page, coords_osr, coords_dzi = self.page_tile_indices[level]
        coord, (w, h) = coords_osr[row][col], coords_dzi[row][col][2:]

        return page, coord, (w, h)

    def get_dzi(self, format=None):
        return get_dzi(
            self.dz_dimensions[-1], 
            tile_size=self.tile_size, 
            overlap=self.overlap, 
            format=format or self.default_format,
        )