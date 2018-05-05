# Universidad del Valle de Guatemala
# Bryan Chan
# 14469

import struct
from collections import namedtuple
from obj import Obj, Texture
import numpy as np

Vertex2 = namedtuple('Vertex2', ['x', 'y'])
Vertex3 = namedtuple('Vertex3', ['x','y','z'])

def char(c):
	return struct.pack('=c', c.encode('ascii'))

def word(w):
	return struct.pack('=h', w)

def dword(d):
	return struct.pack('=l', d)

def color(r, g, b):
	return bytes([b, g, r])

def sumVertex(v0, v1):
	return Vertex3(
		v0.x+v1.x,
		v0.y+v1.y,
		v0.z+v1.z
	)

def subVertex(v0, v1):
	return Vertex3(
		v0.x-v1.x,
		v0.y-v1.y,
		v0.z-v1.z
	)

def multKVertex(v0, k):
	return Vertex3(
		v0.x*k,
		v0.y*k,
		v0.z*k
	)

def dotVertex(v0, v1):
	return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z

def crossVertex(v0, v1):
	return Vertex3(
		v0.y*v1.z - v0.z*v1.y,
		v0.z*v1.x - v0.x*v1.z,
		v0.x*v1.y - v0.y*v1.x
	)

def lenVertex(v):
	return (v.x**2 + v.y**2 + v.z**2)**0.5

def normVertex(v):
	vLenght = lenVertex(v)

	if not vLenght:
		return Vertex3(0,0,0)

	return Vertex3(
		v.x / vLenght,
		v.y / vLenght,
		v.z / vLenght
	)

def getBoundingBox(*vertices):
	xs = [v.x for v in vertices]
	ys = [v.y for v in vertices]

	mins = Vertex2(min(xs), min(ys))
	maxs = Vertex2(max(xs), max(ys))

	return mins, maxs

def barycentric(A, B, C, P):
	bary = crossVertex(
		Vertex3(C.x - A.x, B.x - A.x, A.x - P.x), 
		Vertex3(C.y - A.y, B.y - A.y, A.y - P.y)
	)

	if abs(bary[2]) < 1:
		return Vertex3(-1, -1, -1)

	return Vertex3(
		1 - (bary.x + bary.y)/bary.z,
		(bary.y / bary.z),
		(bary.x / bary.z)
	)


class Render(object):
	def __init__(self, width, height, scalex, scaley, translatex, translatey):
		self.scalex = scalex
		self.scaley = scaley
		self.translatex = translatex
		self.translatey = translatey
		self.width = width
		self.height = height
		self.pixels = []
		self.clear()

	def setBackground(self, pixels):
		self.pixels = pixels

	def clear(self):
		self.pixels = [[color(0,0,0) for x in range(self.width)] for y in range(self.height)]
		self.zbuffer = [[-float('inf') for x in range(self.width)] for y in range(self.height)]

	def write(self, filename):
		f = open(filename, 'wb')

		# Header (14)
		f.write(char('B'))
		f.write(char('M'))
		f.write(dword(14 + 40 + self.width*self.height*3))
		f.write(dword(0))
		f.write(dword(14 + 40))

		# Image header (40)
		f.write(dword(40))
		f.write(dword(self.width))
		f.write(dword(self.height))
		f.write(word(1))
		f.write(word(24))
		f.write(dword(0))
		f.write(dword(self.width*self.height*3))
		f.write(dword(0))
		f.write(dword(0))
		f.write(dword(0))
		f.write(dword(0))


		# Image pixel data (height x width x 3)
		for x in range(self.height):
			for y in range(self.width):
				f.write(self.pixels[x][y])

		f.close()

	def point(self, x, y, color):
		try:
			self.pixels[y][x] = color
		except:
			pass

	def display(self):
		filename = 'out.bmp'
		self.write(filename)

		from PIL import Image

		img = Image.open('out.bmp')
		img.show()

	def oldLine(self, start, end, color):
		x1, y1 = start
		x2, y2 = end

		x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

		dx = abs(x2 - x1)
		dy = abs(y2 - y1)
		m = dy*2

		steep = dy > dx

		if steep:
			x1, y1 = y1, x1
			x2, y2 = y2, x2

		if x2 < x1:
			x1, x2 = x2, x1
			y1, y2 = y2, y1

		dx = x2 - x1
		dy = y2 - y1
		m = dy * 2

		offset = 0
		threshold = dx
		y = y1

		for x in range(x1, x2+1):
			if steep:
				self.point(y,x,color)
			else:
				self.point(x,y,color)
			offset += dy * 2
			if(offset > threshold):
				y += 1 if y1 < y2 else -1
				threshold += 2*dx

	def lowLine(self, start, end, color):
		x0, y0 = start
		x1, y1 = end

		dx = x1 - x0
		dy = y1 - y0
		yi = 1
		if dy < 0:
			yi = -1
			dy = -dy
		D = 2 * dy - dx
		y = y0

		for x in range(x0, x1+1):
			self.point(x, y, color)
			if D > 0:
				y = y + yi
				D = D - 2 * dx
			D = D + 2 * dy

	def highLine(self, start, end, color):
		x0, y0 = start
		x1, y1 = end

		dx = x1 - x0
		dy = y1 - y0
		xi = 1
		if dx < 0:
			xi = -1
			dx = -dx

		D = 2 * dx - dy
		x = x0

		for y in range(y0, y1+1):
			self.point(x, y, color)
			if D > 0:
				x = x + xi
				D = D - 2 * dy
			D = D + 2 * dx

	def line(self, start, end, color):
		x0, y0 = start
		x1, y1 = end
		if abs(y1 - y0) < abs(x1 - x0):
			if x0 > x1:
				self.lowLine(end, start, color)
			else:
				self.lowLine(start, end, color)
		else:
			if y0 > y1:
				self.highLine(end, start, color)
			else:
				self.highLine(start, end, color)

	def lowLinePixels(self, start, end, color):
		x0, y0 = start
		x1, y1 = end

		dx = x1 - x0
		dy = y1 - y0
		yi = 1
		if dy < 0:
			yi = -1
			dy = -dy
		D = 2 * dy - dx
		y = y0
		pixels = []
		for x in range(x0, x1+1):
			# self.point(x, y, color)
			pixels.append((x,y))
			if D > 0:
				y = y + yi
				D = D - 2 * dx
			D = D + 2 * dy
		return pixels

	def highLinePixels(self, start, end, color):
		x0, y0 = start
		x1, y1 = end

		dx = x1 - x0
		dy = y1 - y0
		xi = 1
		if dx < 0:
			xi = -1
			dx = -dx

		D = 2 * dx - dy
		x = x0
		pixels = []
		for y in range(y0, y1+1):
			# self.point(x, y, color)
			pixels.append((x,y))
			if D > 0:
				x = x + xi
				D = D - 2 * dy
			D = D + 2 * dx
		return pixels

	def linePixels(self, start, end, color):
		x0, y0 = start
		x1, y1 = end
		if abs(y1 - y0) < abs(x1 - x0):
			if x0 > x1:
				return self.lowLinePixels(end, start, color)
			else:
				return self.lowLinePixels(start, end, color)
		else:
			if y0 > y1:
				return self.highLinePixels(end, start, color)
			else:
				return self.highLinePixels(start, end, color)

	def oldTriangle(self, v0, v1, v2, color):
		a = Vertex2(*v0)
		b = Vertex2(*v1)
		c = Vertex2(*v2)

		if a.y > b.y:
			a, b = b, a

		if a.y > c.y:
			a, c = c, a

		if b.y > c.y:
			b, c = c, b

		dx_ac = c.x - a.x
		dy_ac = c.y - a.y
		if dy_ac == 0:
			return
		mi_ac = dx_ac/dy_ac

		dx_ab = b.x - a.x
		dy_ab = b.y - a.y
		if dy_ab != 0:
			mi_ab = dx_ab/dy_ab

			for y in range(a.y, b.y + 1):
				xi = round(a.x - mi_ac*(a.y-y))
				xf = round(a.x - mi_ab*(a.y-y))


				if xi > xf:
					xi, xf = xf, xi

				for x in range(xi, xf):
					self.point(x, y, color)

		dx_bc = c.x - b.x
		dy_bc = c.y - b.y
		if dy_bc:
			mi_bc = dx_bc/dy_bc

			for y in range(b.y, c.y + 1):
				xi = round(a.x - mi_ac*(a.y-y))
				xf = round(b.x - mi_bc*(b.y-y))

				if xi > xf:
					xi, xf = xf, xi

				for x in range(xi, xf):
					self.point(x, y, color)

		self.line(a, b, color)
		self.line(b, c, color)
		self.line(c, a, color)

	def triangle(self, A, B, C, color=None, textureCoords=(), varyingNormals=()):
		mins, maxs = getBoundingBox(A, B, C)
		for x in range(mins.x, maxs.x + 1):
			for y in range(mins.y, maxs.y + 1):
				# print("AA")
				# self.point(x, y, color)
				bary = barycentric(A, B, C, Vertex3(x, y, 1))
				if bary.x<0 or bary.y<0 or bary.z<0:
					continue

				# self.point(x, y, color)

				color = self.shader(
					self,
					triangle = (A, B, C),
					bar = (bary.x, bary.y, bary.z),
					varyingNormals = varyingNormals,
					textureCoords = textureCoords)

				z = A.z*bary.x + B.z*bary.y + C.z*bary.z

				if x < len(self.zbuffer) and y < len(self.zbuffer[x]) and z > self.zbuffer[x][y]:
					self.point(x, y, color)
					self.zbuffer[x][y] = z

	def square(self, v0, v1, v2, v3, color):
		a = Vertex2(*v0)
		b = Vertex2(*v1)
		c = Vertex2(*v2)
		d = Vertex2(*v3)

		for i in range(3):
			if a.x > b.x:
				a, b = b, a
			if b.x > c.x:
				b, c = c, b
			if c.x > d.x:
				c, d = d, c

		if(a.y < b.y):
			leftNode = a
			innerNode1 = b
		else:
			leftNode = b
			innerNode1 = a

		if(c.y < d.y):
			rightNode = d
			innerNode2 = c
		else:
			rightNode = c
			innerNode2 = d



		self.oldTriangle(leftNode, rightNode, innerNode1, color)
		self.oldTriangle(leftNode, rightNode, innerNode2, color)

	def transformVertex(self, vertex, translate=Vertex3(0,0,0), scale=Vertex3(1,1,1)):
		augmentedVertex = [
			vertex.x,
			vertex.y,
			vertex.z,
			1
		]

		transformedVertex = np.dot(
			self.ViewPort @ self.Projection @ self.View,
			augmentedVertex
		).tolist()[0]

		return Vertex3(
			round(transformedVertex[0]/transformedVertex[3]),
			round(transformedVertex[1]/transformedVertex[3]),
			round(transformedVertex[2]/transformedVertex[3])
		)

	def load(self, filename, translate=Vertex3(0, 0, 0), scale=Vertex3(1, 1, 1), texture=None, shader=None, normalMap=None):
		"""
		Loads an obj file in the screen
		wireframe only
		Input: 
		  filename: the full path of the obj file
		  translate: (translateX, translateY) how much the model will be translated during render
		  scale: (scaleX, scaleY) how much the model should be scaled
		"""
		model = Obj(filename)
		self.light = normVertex(self.light)
		self.texture = texture
		self.shader = shader
		self.normalMap = normalMap

		for face in model.faces:
			vcount = len(face)

			if vcount == 3:
				f1 = face[0][0] - 1
				f2 = face[1][0] - 1
				f3 = face[2][0] - 1

				# a = self.transform(model.vertices[f1], translate, scale)
				# b = self.transform(model.vertices[f2], translate, scale)
				# c = self.transform(model.vertices[f3], translate, scale)

				a = self.transformVertex(Vertex3(*model.vertices[f1]), translate, scale)
				b = self.transformVertex(Vertex3(*model.vertices[f2]), translate, scale)
				c = self.transformVertex(Vertex3(*model.vertices[f3]), translate, scale)
				# print("a", a)
				# print("b", b)
				# print("c", c)
				# print('face', face)
				n1 = face[0][2] - 1
				n2 = face[1][2] - 1
				n3 = face[2][2] - 1

				nA = Vertex3(*model.normals[n1])
				nB = Vertex3(*model.normals[n2])
				nC = Vertex3(*model.normals[n3])

				t1 = face[0][1] - 1
				t2 = face[1][1] - 1
				t3 = face[2][1] - 1

				# print('model.tvertices', model.tvertices[t1])
				tA = Vertex3(*model.tvertices[t1])
				tB = Vertex3(*model.tvertices[t2])
				tC = Vertex3(*model.tvertices[t3])

				self.triangle(a, b, c, textureCoords=(tA, tB, tC), varyingNormals=(nA, nB, nC))


			elif vcount == 4:
				continue
				# assuming 4
				f1 = face[0][0] - 1
				f2 = face[1][0] - 1
				f3 = face[2][0] - 1
				f4 = face[3][0] - 1

				vertices = [
				  self.transformVertex(model.vertices[f1], translate, scale),
				  self.transformVertex(model.vertices[f2], translate, scale),
				  self.transformVertex(model.vertices[f3], translate, scale),
				  self.transformVertex(model.vertices[f4], translate, scale)
				]

				normal = normVertex(crossVertex(subVertex(vertices[0], vertices[1]), subVertex(vertices[1], vertices[2])))
				intensity = dotVertex(normal, light)
				grey = round(255 * intensity)
				if grey < 0:
					continue # dont paint this face

				A, B, C, D = vertices 

				self.triangle(A, B, C, color(grey, grey, grey))
				self.triangle(A, C, B, color(grey, grey, grey))

	def lookAt(self, eye, center, up):
		z = normVertex(subVertex(eye, center))
		x = normVertex(crossVertex(up, z))
		y = normVertex(crossVertex(z, x))

		self.View = np.matrix([
			[x.x, x.y, x.z, -center.x],
			[y.x, y.y, y.z, -center.y],
			[z.x, z.y, z.z, -center.z],
			[0, 0, 0, 1]
		])

		self.projection(-1 / lenVertex(subVertex(eye, center)))
		self.viewport()

	def projection(self, coeff):
		self.Projection = np.matrix([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, coeff, 1]
		])

	def viewport(self):
		# x-scale, 0, 0, x-translation
		# 0, y-scale, 0, y-translation
		# 0, 0, 128, 128
		# 0, 0, 0, 1
		self.ViewPort = np.matrix([
			[self.scalex, 0, 0, self.translatex],
			[0, self.scaley, 0, self.translatey],
			[0, 0, 128, 128],
			[0, 0, 0, 1]
		])
