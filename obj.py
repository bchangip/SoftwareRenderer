# ===============================================================
# Loads an OBJ file
# ===============================================================
import struct

def color(r, g, b):
  return bytes([b, g, r])


class Obj(object):
  def __init__(self, filename):
    with open(filename) as f:
      self.lines = f.read().splitlines()
    self.vertices = []
    self.tvertices = []
    self.normals = []
    self.faces = []
    self.read()

  def read(self):
    for line in self.lines:
      if line:
        prefix, value = line.split(' ', 1)
        if prefix == 'v':
          self.vertices.append(list(map(float, value.split(' '))))
        elif prefix == 'vt':
          self.tvertices.append(list(map(float, value.split(' '))))
        elif prefix == 'vn':
          self.normals.append(list(map(float, value.split(' '))))
        elif prefix == 'f':
          self.faces.append([list(map(int, face.split('/'))) for face in value.split(' ')])

class Texture(object):
  def __init__(self, path):
    self.path = path
    self.read()

  def read(self):
    image = open(self.path, 'rb')
    image.seek(2 + 4 + 4)
    headerSize = struct.unpack("=l", image.read(4))[0]
    image.seek(2 + 4 + 4 + 4 + 4)

    self.width = struct.unpack("=l", image.read(4))[0]
    self.height = struct.unpack("=l", image.read(4))[0]
    self.pixels = []
    image.seek(headerSize)

    for y in range(self.height):
      self.pixels.append([])
      for x in range(self.width):
        b = ord(image.read(1))
        g = ord(image.read(1))
        r = ord(image.read(1))
        self.pixels[y].append(color(r, g, b))

    image.close()

  def getColor(self, tx, ty):
    x = int(tx * self.width) - 1
    y = int(ty * self.height) - 1
    return self.pixels[y][x]

class NormalMap(object):
  def __init__(self, path):
    self.path = path
    self.read()

  def read(self):
    image = open(self.path, 'rb')
    image.seek(2 + 4 + 4)
    headerSize = struct.unpack("=l", image.read(4))[0]
    image.seek(2 + 4 + 4 + 4 + 4)

    self.width = struct.unpack("=l", image.read(4))[0]
    self.height = struct.unpack("=l", image.read(4))[0]
    self.pixels = []
    image.seek(headerSize)

    for y in range(self.height):
      self.pixels.append([])
      for x in range(self.width):
        nx = ord(image.read(1))
        ny = ord(image.read(1))
        nz = ord(image.read(1))
        self.pixels[y].append([nx, ny, nz])

    image.close()

  def getNormal(self, tx, ty):
    x = int(tx * self.width)
    y = int(ty * self.height)
    color = self.pixels[y][z]
    normal = [0, 0, 0]
    normal[0] = (color[0] / 128) - 1
    normal[1] = (color[1] / 128) - 1
    normal[2] = (color[2] / 256)
    return normal
