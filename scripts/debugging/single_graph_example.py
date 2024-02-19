from visual_encoder.deubuging_tools import img_gen, debugging_estimate_shift, generate_plot_variables

img_0 = img_gen(0,0)
img_1 = img_gen(10,100)
deltay, deltax, variables = debugging_estimate_shift(img_0,img_1)
print(deltax, deltay)
plt = generate_plot_variables(img_0, img_1, variables, "Imagem teste")
plt.show()