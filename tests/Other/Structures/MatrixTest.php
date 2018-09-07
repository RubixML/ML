<?php

namespace Rubix\Tests\Other\Structures;

use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\Other\Structures\Vector;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use IteratorAggregate;
use ArrayAccess;
use Countable;

class MatrixTest extends TestCase
{
    protected $a;

    protected $b;

    protected $c;

    public function setUp()
    {
        $this->a = new Matrix([
            [22, -17, 12],
            [4, 11, -2],
            [20, -6, -9],
        ], true);

        $this->b = new Matrix([
            [13],
            [11],
            [9],
        ], false);

        $this->c =  new Matrix([
            [4, 6, -12],
            [1, 3, 5],
            [-10, -1, 14],
        ], false);
    }

    public function test_build_structure()
    {
        $this->assertInstanceOf(Matrix::class, $this->a);
        $this->assertInstanceOf(Countable::class, $this->a);
        $this->assertInstanceOf(IteratorAggregate::class, $this->a);
        $this->assertInstanceOf(ArrayAccess::class, $this->a);
    }

    public function test_build_identity()
    {
        $d = Matrix::identity(4)->asArray();

        $outcome = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_build_zeros()
    {
        $d = Matrix::zeros(2, 4)->asArray();

        $outcome = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_build_ones()
    {
        $d = Matrix::ones(4, 2)->asArray();

        $outcome = [
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_build_diagonal()
    {
        $d = Matrix::diagonal([0, 1, 4, 5])->asArray();

        $outcome = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 5],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_build_random()
    {
        $d = Matrix::gaussian(4, 4);

        $this->assertCount(16, $d);
    }

    public function test_build_gaussian()
    {
        $d = Matrix::gaussian(3, 3);

        $this->assertCount(9, $d);
    }

    public function test_build_uniform()
    {
        $d = Matrix::uniform(3, 3);

        $this->assertCount(9, $d);
    }

    public function test_minimum()
    {
        $minimum = Matrix::minimum($this->a, $this->c)->asArray();

        $outcome = [
            [4, -17, -12],
            [1, 3, -2],
            [-10, -6, -9],
        ];

        $this->assertEquals($outcome, $minimum);
    }

    public function test_maximum()
    {
        $minimum = Matrix::maximum($this->a, $this->c)->asArray();

        $outcome = [
            [22, 6, 12],
            [4, 11, 5],
            [20, -1, 14],
        ];

        $this->assertEquals($outcome, $minimum);
    }

    public function test_shape()
    {
        $this->assertEquals([3, 3], $this->a->shape());
        $this->assertEquals([3, 1], $this->b->shape());
        $this->assertEquals([3, 3], $this->c->shape());
    }

    public function test_size()
    {
        $this->assertEquals(9, $this->a->size());
        $this->assertEquals(3, $this->b->size());
        $this->assertEquals(9, $this->c->size());
    }

    public function test_get_m()
    {
        $this->assertEquals(3, $this->a->m());
        $this->assertEquals(3, $this->b->m());
        $this->assertEquals(3, $this->c->m());
    }

    public function test_get_n()
    {
        $this->assertEquals(3, $this->a->n());
        $this->assertEquals(1, $this->b->n());
        $this->assertEquals(3, $this->c->n());
    }

    public function test_get_row()
    {
        $this->assertEquals([22, -17, 12], $this->a->row(0));
        $this->assertEquals([11], $this->b->row(1));
        $this->assertEquals([-10, -1, 14], $this->c->row(2));
    }

    public function test_get_column()
    {
        $this->assertEquals([-17, 11, -6], $this->a->column(1));
        $this->assertEquals([13, 11, 9], $this->b->column(0));
        $this->assertEquals([-12, 5, 14], $this->c->column(2));
    }

    public function test_as_array()
    {
        $outcome = [
            [22, -17, 12],
            [4, 11, -2],
            [20, -6, -9],
        ];

        $this->assertEquals($outcome, $this->a->asArray());
    }

    public function test_as_vectors()
    {
        $vectors = $this->a->asVectors();

        $this->assertEquals([22, -17, 12], $vectors[0]->asArray());
        $this->assertEquals([4, 11, -2], $vectors[1]->asArray());
        $this->assertEquals([20, -6, -9], $vectors[2]->asArray());
    }

    public function test_transpose()
    {
        $d = $this->a->transpose()->asArray();

        $outcome = [
            [22, 4, 20],
            [-17, 11, -6],
            [12, -2, -9],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_map()
    {
        $d = $this->a->map(function ($value) {
            return $value > 0. ? 1 : -1;
        });

        $outcome = [
            [1, -1, 1],
            [1, 1, -1],
            [1, -1, -1],
        ];

        $this->assertEquals($outcome, $d->asArray());
    }

    public function test_reduce()
    {
        $scalar = $this->a->reduce(function ($value, $carry) {
            return $carry + ($value / 2.);
        });

        $this->assertEquals(17.5, $scalar);
    }

    public function test_dot()
    {
        $d = $this->a->dot($this->b)->asArray();

        $outcome = [
            [207], [155], [113],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_multiply()
    {
        $d = $this->a->multiply($this->c)->asArray();

        $outcome = [
            [88, -102, -144],
            [4, 33, -10],
            [-200, 6, -126],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_divide()
    {
        $d = $this->a->divide($this->c)->asArray();

        $outcome = [
            [5.5, -2.8333333333333335, -1],
            [4, 3.6666666666666665, -0.4],
            [-2, 6, -0.6428571428571429],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_add()
    {
        $d = $this->a->add($this->c)->asArray();

        $outcome = [
            [26, -11, 0],
            [5, 14, 3],
            [10, -7, 5],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_subtract()
    {
        $d = $this->a->subtract($this->c)->asArray();

        $outcome = [
            [18, -23, 24],
            [3, 8, -7],
            [30, -5, -23],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_reciprocal()
    {
        $d = $this->a->reciprocal()->asArray();

        $outcome = [
            [0.045454545454545456, -0.058823529411764705, 0.08333333333333333],
            [0.25, 0.09090909090909091, -0.5],
            [0.05, -0.16666666666666666, -0.1111111111111111],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_power()
    {
        $d = $this->a->power($this->c)->asArray();

        $outcome = [
            [234256, 24137569, 1.1215665478461509E-13],
            [4, 1331, -32],
            [9.765625E-14, -0.16666666666666666, 22876792454961],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_scalar_multiply()
    {
        $d = $this->a->multiplyScalar(2.5)->asArray();

        $outcome = [
            [55, -42.5, 30],
            [10., 27.5, -5.],
            [50, -15, -22.5],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_scalar_divide()
    {
        $d = $this->a->divideScalar(2.)->asArray();

        $outcome = [
            [11., -8.5, 6.],
            [2., 5.5, -1.],
            [10., -3., -4.5],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_scalar_add()
    {
        $d = $this->a->addScalar(1)->asArray();

        $outcome = [
            [23, -16, 13],
            [5, 12, -1],
            [21, -5, -8],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_scalar_subtract()
    {
        $d = $this->a->subtractScalar(10)->asArray();

        $outcome = [
            [12, -27, 2],
            [-6, 1, -12],
            [10, -16, -19],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_sum()
    {
        $this->assertEquals([17, 13, 5], $this->a->sum()->asArray());
    }

    public function test_row_sum()
    {
        $this->assertEquals(17., $this->a->rowSum(0));
        $this->assertEquals(11., $this->b->rowSum(1));
        $this->assertEquals(3., $this->c->rowSum(2));
    }

    public function test_column_sum()
    {
        $this->assertEquals(46., $this->a->columnSum(0));
        $this->assertEquals(0., $this->b->columnSum(1));
        $this->assertEquals(7., $this->c->columnSum(2));
    }

    public function test_product()
    {
        $this->assertEquals([-4488., -88., 1080.], $this->a->product()->asArray());
    }

    public function test_abs()
    {
        $d = $this->a->abs()->asArray();

        $outcome = [
            [22, 17, 12],
            [4, 11, 2],
            [20, 6, 9],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_square()
    {
        $d = $this->a->square()->asArray();

        $outcome = [
            [484, 289, 144],
            [16, 121, 4],
            [400, 36, 81],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_square_root()
    {
        $d = $this->b->sqrt()->asArray();

        $outcome = [
            [3.605551275463989],
            [3.3166247903554],
            [3],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_exp()
    {
        $d = $this->b->exp()->asArray();

        $outcome = [
            [442413.3920089202],
            [59874.14171519778],
            [8103.08392757538],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_log()
    {
        $d = $this->b->log()->asArray();

        $outcome = [
            [2.5649493574615367],
            [2.3978952727983707],
            [2.1972245773362196],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_mean()
    {
        $d = $this->a->mean()->asArray();

        $outcome = [5.666666666666667, 4.333333333333333, 1.6666666666666667];

        $this->assertEquals($outcome, $d);
    }

    public function test_covariance()
    {
        $mean = $this->a->transpose()->mean();

        $d = $this->a->covariance($mean)->asArray();

        $outcome = [
            [116.51851851851852, -99.25925925925925, -17.259259259259263],
            [-99.25925925925925, 119.62962962962963, -20.370370370370367],
            [-17.259259259259263, -20.370370370370367, 37.62962962962963],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_round()
    {
        $d = $this->a->round(2)->asArray();

        $outcome = [
            [22, -17, 12],
            [4, 11, -2],
            [20, -6, -9],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_floor()
    {
        $d = $this->a->floor()->asArray();

        $outcome = [
            [22, -17, 12],
            [4, 11, -2],
            [20, -6, -9],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_ceil()
    {
        $d = $this->a->ceil()->asArray();

        $outcome = [
            [22, -17, 12],
            [4, 11, -2],
            [20, -6, -9],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_l1_norm()
    {
        $this->assertEquals(46., $this->a->l1Norm());
        $this->assertEquals(33., $this->b->l1Norm());
        $this->assertEquals(31., $this->c->l1Norm());
    }

    public function test_l2_norm()
    {
        $this->assertEquals(39.68626966596886, $this->a->l2Norm());
        $this->assertEquals(19.261360284258224, $this->b->l2Norm());
        $this->assertEquals(22.978250586152114, $this->c->l2Norm());
    }

    public function test_infinity_norm()
    {
        $this->assertEquals(51.0, $this->a->infinityNorm());
        $this->assertEquals(13.0, $this->b->infinityNorm());
        $this->assertEquals(25.0, $this->c->infinityNorm());
    }

    public function test_max_norm()
    {
        $this->assertEquals(22.0, $this->a->maxNorm());
        $this->assertEquals(13.0, $this->b->maxNorm());
        $this->assertEquals(14.0, $this->c->maxNorm());
    }

    public function test_binarize()
    {
        $outcome = [
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 0],
        ];

        $this->assertEquals($outcome, $this->a->binarize()->asArray());
    }

    public function test_negate()
    {
        $outcome = [
            [-22, 17, -12],
            [-4, -11, 2],
            [-20, 6, 9],
        ];

        $this->assertEquals($outcome, $this->a->negate()->asArray());
    }

    public function test_row_exclude()
    {
        $d = $this->a->rowExclude(2)->asArray();

        $outcome = [
            [22, -17, 12],
            [4, 11, -2],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_column_exclude()
    {
        $d = $this->a->columnExclude(1)->asArray();

        $outcome = [
            [22, 12],
            [4, -2],
            [20, -9],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_augment_above()
    {
        $d = $this->a->augmentAbove($this->c)->asArray();

        $outcome = [
            [4, 6, -12],
            [1, 3, 5],
            [-10, -1, 14],
            [22, -17, 12],
            [4, 11, -2],
            [20, -6, -9],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_augment_below()
    {
        $d = $this->a->augmentBelow($this->c)->asArray();

        $outcome = [
            [22, -17, 12],
            [4, 11, -2],
            [20, -6, -9],
            [4, 6, -12],
            [1, 3, 5],
            [-10, -1, 14],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_augment_left()
    {
        $d = $this->a->augmentLeft($this->b)->asArray();

        $outcome = [
            [13, 22, -17, 12],
            [11, 4, 11, -2],
            [9, 20, -6, -9],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_augment_right()
    {
        $d = $this->a->augmentRight($this->b)->asArray();

        $outcome = [
            [22, -17, 12, 13],
            [4, 11, -2, 11],
            [20, -6, -9, 9],
        ];

        $this->assertEquals($outcome, $d);
    }

    public function test_repeat()
    {
        $d = $this->b->repeat(2, 4)->asArray();

        $outcome = [
            [13, 13, 13, 13],
            [11, 11, 11, 11],
            [9, 9, 9, 9],
            [13, 13, 13, 13],
            [11, 11, 11, 11],
            [9, 9, 9, 9],
        ];

        $this->assertEquals($outcome, $d);
    }
}
