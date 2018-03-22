<?php

use Rubix\Engine\Math\Matrix;
use PHPUnit\Framework\TestCase;

class MatrixTest extends TestCase
{
    protected $matrix;

    public function setUp()
    {
        $this->matrix = new Matrix([
            [1, -4],
            [-3, 6],
        ]);
    }

    public function test_build_matrix()
    {
        $this->assertTrue($this->matrix instanceof Matrix);
    }

    public function test_build_column_vector()
    {
        $vector = Matrix::vector([2, 1, 6]);

        $this->assertTrue($vector instanceof Matrix);
        $this->assertEquals([[2], [1], [6]], $vector->values());
    }

    public function test_build_row_vector()
    {
        $vector = Matrix::vector([2, 1, 6], 'row');

        $this->assertTrue($vector instanceof Matrix);
        $this->assertEquals([[2, 1, 6]], $vector->values());
    }

    public function test_calculate_dot_product()
    {
        $matrix = Matrix::vector([4, 1, 6])->dot(Matrix::vector([9, 5, 2]));

        $answer = [
            [36, 20, 8],
            [9, 5, 2],
            [54, 30, 12],
        ];

        $this->assertTrue($matrix instanceof Matrix);
        $this->assertEquals($answer, $matrix->values());
    }

    public function test_select_row()
    {
        $this->assertEquals([1, -4], $this->matrix->row(0)->values()[0]);
        $this->assertEquals([-3, 6], $this->matrix->row(1)->values()[0]);
    }

    public function test_select_column()
    {
        $this->assertEquals([1, -3], $this->matrix->column(0)->values()[0]);
        $this->assertEquals([-4, 6], $this->matrix->column(1)->values()[0]);
    }

    public function test_get_identity_matrix()
    {
        $answer = [
            [1, 0],
            [0, 1],
        ];

        $this->assertEquals($answer, $this->matrix->identity()->values());
    }

    public function test_get_dimensions()
    {
        $this->assertEquals([2, 2], $this->matrix->dimensions());
    }

    public function test_get_rows()
    {
        $this->assertEquals(2, $this->matrix->rows());
    }

    public function test_get_columns()
    {
        $this->assertEquals(2, $this->matrix->columns());
    }

    public function test_add_matrix()
    {
        $matrix = new Matrix([
            [9, 2],
            [1, 8],
        ]);

        $answer = [
            [10, -2],
            [-2, 14],
        ];

        $this->assertEquals($answer, $this->matrix->add($matrix)->values());
    }

    public function test_subtract_matrix()
    {
        $matrix = new Matrix([
            [-3, 2],
            [5, -8],
        ]);

        $answer = [
            [4, -6],
            [-8, 14],
        ];

        $this->assertEquals($answer, $this->matrix->subtract($matrix)->values());
    }

    public function test_multiply_matrix()
    {
        $matrix = new Matrix([
            [3, 1],
            [7, 2],
        ]);

        $answer = [
            [-25, -7],
            [33, 9],
        ];

        $this->assertEquals($answer, $this->matrix->multiply($matrix)->values());
    }

    public function test_multiply_by_scalar()
    {
        $answer = [
            [2, -8],
            [-6, 12],
        ];

        $this->assertEquals($answer, $this->matrix->multiplyByScalar(2)->values());
    }

    public function test_divide_by_scalar()
    {
        $answer = [
            [0.5, -2],
            [-1.5, 3],
        ];

        $this->assertEquals($answer, $this->matrix->divideByScalar(2)->values());
    }
}
