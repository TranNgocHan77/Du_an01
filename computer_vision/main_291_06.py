from PIL import Image
from PIL.ImageOps import grayscale

def falsecolor(src_path, colors):
    # Check if the source is a file path, then open the image
    if isinstance(src_path, str):
        src = Image.open(src_path)
    else:
        raise TypeError('Invalid source type. Provide a file path.')

    # Check if the image mode is supported
    if src.mode not in ['L', 'RGB', 'RGBA']:
        raise TypeError(f'Unsupported source image mode: {src.mode}')

    src.load()

    # Create look-up-tables (LUTs) to map luminosity ranges to color components
    num_colors = len(colors)
    palette = [colors[int(i / 256. * num_colors)] for i in range(256)]
    luts = tuple(c[channel] for channel in range(3) for c in palette)

    # Create grayscale version of the image if necessary
    l = src if src.mode == 'L' else grayscale(src)

    # Convert grayscale to an equivalent RGB mode image
    if Image.getmodebands(src.mode) < 4:  # Non-alpha image
        merge_args = ('RGB', (l,) * 3)  # RGB version of grayscale
    else:  # Include a copy of the source image's alpha layer
        a = Image.new('L', src.size)
        a.putdata(src.getdata(3))
        luts += tuple(range(256))  # Add a 1:1 mapping for alpha values
        merge_args = ('RGBA', (l,) * 3 + (a,))  # RGBA version of grayscale

    # Merge all the grayscale bands back together and apply the LUTs
    return Image.merge(*merge_args).point(luts)

if __name__ == '__main__':
    # Define color constants
    R, G, B = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    C, M, Y = (0, 255, 255), (255, 0, 255), (255, 255, 0)

    # Specify the image file path
    filename = 'UTC2_01_trang_den.jpg'

    # Convert the image into a false-color one with 4 colors and display it
    falsecolor_result = falsecolor(filename, [B, R, G, Y])
    falsecolor_result.show()
