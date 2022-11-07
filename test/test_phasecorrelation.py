import matplotlib.pyplot as plt
from visual_encoder.phase_correlation import *
from visual_encoder import shift_estimators
from PIL import Image
myImage = Image.open("/home/tekalid/repos/AUSPEX/test/visual_encoder/data/images/rgb_example.jpeg");
img_rgb = np.array(myImage)

# Descolamento desejado:
xshift = 600
yshift = 300

# Conversão RGB -> Grey
img = img_rgb[:, :, 0] * .299 + img_rgb[:, :, 1] * .587 + img_rgb[:, :, 2] * .114

# Autocontraste:
img = np.round(img / img.max() * 255)

# FFTs das imagens:
img2 = img[:1500, :1500]
img1 = img[xshift:xshift + 1500, yshift:yshift + 1500]



# Filtra o ruído numérico:
#
# #
# #
# u, s, vh = np.linalg.svd(phasecorrelation)
# u[:, 0]
# spec_phase = np.angle(phasecorrelation)
# plt.plot(s)
# s[1:] = 0
# s_diag = np.diag(s)
# mult = u @ s_diag @ vh
# plt.imshow(mult)
# plt.plot(mult[750, :])


# Plot:
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagem")

plt.subplot(1, 4, 2)
plt.imshow(img1, cmap='gray')
plt.title("Imagem Original")

plt.subplot(1, 4, 3)
plt.imshow(img2, cmap='gray')
plt.title("Imagem Deslocada")

plt.subplot(1, 4, 4)
cross_correlation, _ = phasecorrelation(img1, img2)
deltax, deltay = shift_estimators.abs_maximum_method(cross_correlation)
plt.imshow(np.log10(cross_correlation + 0.01), cmap='gray')
# plt.colorbar()
plt.title(f"Phase correlation. $(\Delta x, \Delta y) = ({deltax[0]}, {deltay[0]})$")