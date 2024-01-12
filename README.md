# Sudoku Solver and Recognizer

## Project Overview

This project focuses on the recognition and resolution of Sudoku grids using advanced computer vision techniques. It aims to encourage students to apply their knowledge to practical projects and explore advanced areas of computer vision.

![example1](https://github.com/RimaBzch/sudoku-solver-opencv-python/assets/86674923/f4aaef75-19b5-43fb-867f-d3acec99962d)
![example2](https://github.com/RimaBzch/sudoku-solver-opencv-python/assets/86674923/efe7e8da-ff0b-48ca-b22f-93aec31583da)

### 1. Description of Sudoku Game

Sudoku is a grid-based game consisting of 81 cells, inspired by the Latin square and the 36 officers' problem of the Swiss mathematician Leonhard Euler. The objective is to fill the grid with a series of different numbers, ensuring each number appears only once in each row, column, and 3x3 sub-grid.

Three rules must be followed:
- Each row must contain all numbers from 1 to 9.
- Each column must contain all numbers from 1 to 9.
- Each 3x3 sub-grid must contain all numbers from 1 to 9.

Initially, some grid cells contain numbers known as indices, forming the basis of the problem. These indices are not randomly chosen and serve as the starting point for a solution.

**Example:** The grid may have 28 indices. For instance, the cell at coordinates (2,6) can only contain the number 2 since it must belong to that region, and the other 5 free cells are restricted. Once the number 2 is revealed, it is forbidden in the second row.

The symbols represent numbers from 1 to 9, and the sub-grids are 3x3 squares. Grid resolution involves a brute-force method testing all possible solutions, providing good results with a computation time of less than half a second for the most challenging grids.

### 2. Assumptions

Users of such a system are unpredictable and may take photos with various irregularities. To streamline the recognition of numbers, the following assumptions are made:
- Grids are initially empty; users take a photo before annotating.
- Sudoku grids should occupy most of the image with minimal surrounding noise.
- Images may have different types of noise and distortions, reasonable enough for grid and number visibility.

### 3. Project Steps

The project involves four main steps:

1. **Normalization**
2. **Grid Detection**
3. **Number Recognition**
4. **Sudoku Resolution**

Feel free to explore and contribute to this project as we delve into the exciting world of Sudoku recognition and solving.
