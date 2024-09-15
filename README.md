## III. Plot Module Features

### A. Focus on using Plotly as a foundation for both R and Python plots
1. Provide low-level access for customization of individual plots
2. All plots should support annotations
   - CBBI logo watermark (location and graphic TBD)
   - Arbitrary text and shape annotations, attached to points with optimal spacing to avoid overlapping with other points and annotations
   - Replace axis number values with wedge to indicate increasing/decreasing values
3. All plot points should support optional hyperlinks to web pages
4. All plot axes should support pseudolog scaling
5. All plots should support subplot panels with titles
6. All plots should support animations along the axis (e.g., time)

### B. Global configuration default options (can be overridden by individual plots)
1. Figure size and dimensions
2. Text color and font size
3. Point color, size, shape, opacity
4. Legend position, size, color bar
5. General color palettes
   - Hot-cold: continuous red-white-blue; neutral point configurable
   - Color deficiency: continuous grayscale
   - Maximum contrast: [https://gist.github.com/Myndex/997244b95d84788df96f4aab8b9edeb1](https://gist.github.com/Myndex/997244b95d84788df96f4aab8b9edeb1)
   - BPMC
     - 006E96
     - 00263D
     - B8D87A
     - 8E847A
     - FFFFFF
     - EB6852
     - 983E53

### C. Support the following plot types
1. 2D scatter
   - Input: Data frame (long and wide formats)
   - Options
     - Color label
     - Shape label
     - Opacity label
     - Individual point label
     - Linear regression with visible
       - Regression line
       - Coefficient of determination
       - Pearson coefficient
       - Confidence intervals
   - Note: t-SNE, UMAP, and PCA are special types of scatter plots that may build upon this foundation
![2D](https://github.com/user-attachments/assets/7f2c04bf-bca2-4af0-9a2c-bf6a4c6f7792)

2. 3D scatter
   - Input: Data frame (long and wide formats)
   - Options
     - Color label
     - Shape label
     - Individual point label
     - Scale point size and opacity to indicate depth
![3D](https://github.com/user-attachments/assets/83816a42-c92c-418b-8b95-014ea56d3813)

3. Line
   - Input: Data frame (long and wide formats)
   - Options
     - Color label
     - Shape label
     - Opacity label
     - Error bars
     - Smoothing
   - Note: AUC, ROC, and VAF are special types of line plots that may build upon this foundation
![line plot](https://github.com/user-attachments/assets/00ec1bf3-0987-4612-a631-202932866b03)

