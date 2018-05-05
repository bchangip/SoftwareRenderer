# Face render
from gl import *

def gourad(render, bar, **kwargs):
  w, v, u = bar

  tA, tB, tC = kwargs['textureCoords']
  tx = tA.x * w + tB.x * v + tC.x * u
  ty = tA.y * w + tB.y * v + tC.y * u
  color = render.texture.getColor(tx, ty)

  iA, iB, iC = [ dotVertex(n, render.light) for n in kwargs['varyingNormals'] ]
  intensity = w * iA + v * iB + u * iC

  return bytes(map(lambda x: round(x * intensity) if x * intensity > 0 else 0, color))

# Stars
starsR = Render(800, 600, 500, 500, 200, 350)
starsR.light = Vertex3(200, 100, 10)
starsT = Texture('stars.bmp')
starsR.lookAt(Vertex3(20, 10, -20), Vertex3(0, 0, 0), Vertex3(0, 1, 0))
starsR.load('earth.obj', texture=starsT, shader=gourad, translate=Vertex3(0,0,0))

# starsR.write('starsTest.bmp')

# Sun
sunR = Render(800, 600, 200, 200, 200, 350)
sunR.setBackground(starsR.pixels)
sunR.light = Vertex3(200, 100, 10)
sunT = Texture('sun.bmp')
sunR.lookAt(Vertex3(20, 10, -20), Vertex3(0, 0, 0), Vertex3(0, 1, 0))
sunR.load('earth.obj', texture=sunT, shader=gourad, translate=Vertex3(0,0,0))

# Earth
earthR = Render(800, 600, 160, 160, 350, 300)
earthR.setBackground(sunR.pixels)
earthR.light = Vertex3(200, 100, 10)
earthT = Texture('earth.bmp')
earthR.lookAt(Vertex3(20, 10, -20), Vertex3(0, 0, 0), Vertex3(0, 1, 0))
earthR.load('earth.obj', texture=earthT, shader=gourad, translate=Vertex3(0,0,0))

# Moon
moonR = Render(800, 600, 90, 90, 500, 300)
moonR.setBackground(earthR.pixels)
moonR.light = Vertex3(200, 100, 10)
moonT = Texture('moon.bmp')
moonR.lookAt(Vertex3(20, 10, -20), Vertex3(0, 0, 0), Vertex3(0, 1, 0))
moonR.load('earth.obj', texture=moonT, shader=gourad, translate=Vertex3(0,0,0))

# Ship
shipR = Render(800, 600, 50, 50, 300, 100)
shipR.setBackground(moonR.pixels)
shipR.light = Vertex3(200, 100, 10)
shipT = Texture('simpleShip.bmp')
shipR.lookAt(Vertex3(20, 10, -20), Vertex3(0, 0, 0), Vertex3(0, 1, 0))
shipR.load('simpleShip.obj', texture=shipT, shader=gourad, translate=Vertex3(0,0,0))

# Sputnik
sputnikR = Render(800, 600, 4000, 4000, 400, 400)
sputnikR.setBackground(shipR.pixels)
sputnikR.light = Vertex3(200, 100, 10)
sputnikT = Texture('sputnik.bmp')
sputnikR.lookAt(Vertex3(20, 10, -20), Vertex3(0, 0, 0), Vertex3(0, 1, 0))
sputnikR.load('sputnik.obj', texture=sputnikT, shader=gourad, translate=Vertex3(0,0,0))

sputnikR.write('solarSistem.bmp')