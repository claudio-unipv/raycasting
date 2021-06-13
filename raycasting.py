#!/usr/bin/env python3

import numpy as np


def make_map(z=0.5, a=np.pi / 2):
    """Create a map with different kinds of walls."""
    data = """
    1111111111
    1........1
    1...*....1
    1......111
    1........1
    1...22...1
    1...22...1
    333......1
    4.3......1
    4.3.....11
    4.....1111
    4444444444
    """.split()
    # The asterisk is the starting position.
    for y, row in enumerate(data):
        if "*" in row:
            x = row.index("*")
            break
    data = [row.replace(".", "0").replace("*", "0") for row in data]
    indices = [list(map(int, row)) for row in data]
    return np.array(indices, dtype=int), x, y, z, a


def make_textures():
    """Compute a 4D array representing a set of texture images."""
    # Create five 256x256 color patterns.
    a = np.arange(256)[None, :]
    textures = 0.5 * np.ones((5, a.size, a.size, 3))
    textures[1, :, :, 1] = (a // 16 + a.T // 16) % 2
    textures[2, :, :, 2] = ((a + a.T) // 16) % 2
    textures[3, :, :, 0] = ((a + 0 * a.T) // 16) % 2
    textures[4, :, :, 0] = ((0 * a + a.T) // 16) % 2
    textures[4, :, :, 1] = ((0 * a + a.T) // 16) % 2
    return textures


def intersect_lines(x, y, as_, walls):
    """Intersect rays with a set of parallel walls.

    Return:
    - rs: distances of the intersection
    - ws: label representing the type of the wall
    - ts: horizontal coordinate of the intersection inside the wall
    """
    #            as_[i]
    #            /
    # ----------/---- ys[0]
    #          /
    # --------/------ ys[1]
    #        /
    # ------/-------- ys[2]
    #      * (x, y)
    # --------------- ys[3]
    #                 ...
    ys = np.arange(1, walls.shape[0])[:, None]
    # Find candidate intersections along lines of constant y
    rs = (ys - y) / np.sin(as_[None, :])
    xs = x + (ys - y) / np.tan(as_[None, :])
    # Set rs to inf where there are no walls
    is_ = np.tile((ys - (y > ys)), (1, xs.shape[1]))
    js = np.clip(xs.astype(int), 0, walls.shape[1] - 1)
    rs[walls[is_, js] == 0] = np.inf
    # Set to inf also if in the wrong direction
    rs[rs <= 0] = np.inf
    # Take as real intersections the ones with the closest wall.
    ks = rs.argmin(0)
    us = np.arange(ks.size)
    xs = xs[ks, us]
    rs = rs[ks, us]
    ws = walls[is_[ks, us], js[ks, us]]
    return rs, ws, np.modf(xs)[0]


def raycast(x, y, as_, walls):
    """Find intersections between rays from x, y into walls.

    as_ are the angles representing the orientations of the rays.

    Return:
    - rs: distances of the intersection
    - ws: label representing the type of the wall
    - ts: horizontal coordinate of the intersection inside the wall

    """
    # intersections with horizontal lines
    rs_h, ws_h, ts_h = intersect_lines(x, y, as_, walls)
    # intersections with vertical lines
    rs_v, ws_v, ts_v = intersect_lines(y, x, np.pi / 2 - as_, walls.T)
    # select among horizontal and vertical
    rs = np.minimum(rs_h, rs_v)
    ts = np.where(rs_h < rs_v, ts_h, ts_v)
    ws = np.where(rs_h < rs_v, ws_h, ws_v)
    return rs, ws, ts


def background_image(height, width, z, floor=(0.4, 0.4, 0.4),
                     ceiling=(0.6, 0.6, 0.6)):
    """Create a background image with floor and ceiling."""
    image = np.empty((height, width, 3))
    image[:, :, :] = np.array(floor).reshape(1, 1, 3)
    h = int((height - 1) * z)
    image[:-h, :, :] = np.array(ceiling).reshape(1, 1, 3)
    return image


def render_line(texture, x, top, bottom, height):
    """Render a vertical line from a texture."""
    column = int(x * (texture.shape[1] - 1))
    line = np.empty((top - bottom, 3))
    for c in range(3):
        y = np.linspace(0, 1, top - bottom)
        yp = np.linspace(0, 1, texture.shape[0])
        cp = texture[np.arange(texture.shape[0]), column, c]
        line[:, c] = np.interp(y, yp, cp)
    if bottom < 0:
        line = line[-bottom:]
    if top >= height:
        line = line[:line.shape[0] - (top - height)]
    return line


def render_image(walls, x, y, z, a, textures, width=512, height=512,
                 fov=np.pi / 4):
    """Create an image for the given map, position and textures."""
    f = np.tan(fov / 2)
    as_ = a + np.arctan(np.linspace(-f, f, width))
    rs, ws, ts = raycast(x, y, as_, walls)
    distance = 1 / (2 * np.tan(fov / 2))
    top = (1 - z) + distance * z / rs
    bottom = (1 - z) - distance * (1 - z) / rs
    # This are needed to prevent out of memory errors when very close
    # to a wall.
    top = np.minimum(10, top)
    bottom = np.maximum(-10, bottom)
    top = (top * (height - 1)).astype(int)
    bottom = (bottom * (height - 1)).astype(int)
    image = background_image(height, width, z)
    for i in range(width):
        line = render_line(textures[ws[i]], ts[i], top[i], bottom[i], height)
        b = max(0, bottom[i])
        t = min(top[i], height)
        image[b:t, i, :] = line
    return image


def main():
    print("""
Keyboard Commands:

    wasd  move arond
      ,.  turn left/right
       q  exit
""")
    walls, x, y, z, a = make_map()
    textures = make_textures()
    dirs = {"w": 0.0, "a": -np.pi / 2, "s": np.pi, "d": np.pi / 2}
    # Main loop: draw the image and wait for a keypress.
    while True:
        image = render_image(walls, x, y, z, a, textures)
        cv2.imshow("scene", image)
        key = chr(cv2.waitKey(0)).lower()
        if key == "q":
            break
        elif key in dirs:
            x += 0.1 * np.cos(a + dirs[key])
            y += 0.1 * np.sin(a + dirs[key])
        elif key == ",":
            a -= np.pi / 20
        elif key == ".":
            a += np.pi / 20
    print("bye")


if __name__ == "__main__":
    import cv2
    main()
