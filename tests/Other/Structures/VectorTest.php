<?php

namespace Rubix\Tests\Other\Structures;

use Rubix\ML\Other\Structures\Vector;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use IteratorAggregate;
use ArrayAccess;
use Countable;

class VectorTest extends TestCase
{
    protected $a;

    protected $b;

    protected $c;

    public function setUp()
    {
        $this->a = new Vector([-15, 25, 35, -36, -72, 89, 106, 45], true);

        $this->b = new Vector([0.25, 0.1, 2., -0.5, -1., -3.0, 3.3, 2.0], false);

        $this->c = new Vector([4., 6.5, 2.9, 20., 2.6, 11.9]);
    }

    public function test_build_structure()
    {
        $this->assertInstanceOf(Vector::class, $this->a);
        $this->assertInstanceOf(Countable::class, $this->a);
        $this->assertInstanceOf(IteratorAggregate::class, $this->a);
        $this->assertInstanceOf(ArrayAccess::class, $this->a);
    }

    public function test_build_zeros()
    {
        $this->assertEquals([0, 0, 0, 0], Vector::zeros(4)->asArray());
    }

    public function test_build_ones()
    {
        $this->assertEquals([1, 1, 1, 1], Vector::ones(4)->asArray());
    }

    public function test_get_n()
    {
        $this->assertEquals(8, $this->a->n());
    }

    public function test_as_array()
    {
        $outcome = [-15, 25, 35, -36, -72, 89, 106, 45];

        $this->assertEquals($outcome, $this->a->asArray());
    }

    public function test_as_row_matrix()
    {
        $outcome = [[-15, 25, 35, -36, -72, 89, 106, 45]];

        $this->assertEquals($outcome, $this->a->asRowMatrix()->asArray());
    }

    public function test_as_column_matrix()
    {
        $outcome = [[-15], [25], [35], [-36], [-72], [89], [106], [45]];

        $this->assertEquals($outcome, $this->a->asColumnMatrix()->asArray());
    }

    public function test_map()
    {
        $c = $this->a->map(function ($value) {
            return $value > 0. ? 1 : 0;
        });

        $outcome = [0, 1, 1, 0, 0, 1, 1, 1];

        $this->assertEquals($outcome, $c->asArray());
    }

    public function test_reduce()
    {
        $scalar = $this->a->reduce(function ($value, $carry) {
            return $carry + ($value / 2.);
        });

        $this->assertEquals(110.3671875, $scalar);
    }

    public function test_dot()
    {
        $this->assertEquals(331.54999999999995, $this->a->dot($this->b));
    }

    public function test_outer()
    {
        $outcome = [
            [-3.75, -1.5, -30., 7.5, 15., 45., -49.5, -30.],
            [6.25, 2.5, 50., -12.5, -25., -75., 82.5, 50.],
            [8.75, 3.5, 70., -17.5, -35., -105., 115.5, 70.],
            [-9.0, -3.6, -72., 18., 36., 108., -118.8, -72.],
            [-18., -7.2, -144., 36., 72., 216., -237.6, -144.],
            [22.25, 8.9, 178., -44.5, -89., -267., 293.7, 178.],
            [26.5, 10.600000000000001, 212., -53., -106., -318., 349.79999999999995, 212.],
            [11.25, 4.5, 90., -22.5, -45., -135., 148.5, 90.],
        ];

        $this->assertEquals($outcome, $this->a->outer($this->b)->asArray());
    }

    public function test_multiply()
    {
        $c = $this->a->multiply($this->b);

        $outcome = [-3.75, 2.5, 70., 18., 72., -267., 349.79999999999995, 90.];

        $this->assertEquals($outcome, $c->asArray());
    }

    public function test_divide()
    {
        $c = $this->a->divide($this->b);

        $outcome = [-60., 250., 17.5, 72., 72., -29.666666666666668, 32.121212121212125, 22.5];

        $this->assertEquals($outcome, $c->asArray());
    }

    public function test_add()
    {
        $c = $this->a->add($this->b);

        $outcome = [-14.75, 25.1, 37., -36.5, -73., 86., 109.3, 47.];

        $this->assertEquals($outcome, $c->asArray());
    }

    public function test_subtract()
    {
        $c = $this->a->subtract($this->b);

        $outcome = [-15.25, 24.9, 33., -35.5, -71., 92., 102.7, 43.];

        $this->assertEquals($outcome, $c->asArray());
    }

    public function test_scalar_multiply()
    {
        $outcome = [-30, 50, 70, -72, -144, 178, 212, 90];

        $this->assertEquals($outcome, $this->a->multiplyScalar(2)->asArray());
    }

    public function test_scalar_divide()
    {
        $outcome = [-7.5, 12.5, 17.5, -18, -36, 44.5, 53, 22.5];

        $this->assertEquals($outcome, $this->a->divideScalar(2)->asArray());
    }

    public function test_scalar_add()
    {
        $outcome = [-5, 35, 45, -26, -62, 99, 116, 55];

        $this->assertEquals($outcome, $this->a->addScalar(10)->asArray());
    }

    public function test_sum()
    {
        $this->assertEquals(177., $this->a->sum());
    }

    public function test_product()
    {
        $this->assertEquals(-14442510600000.0, $this->a->product());
    }

    public function test_abs()
    {
        $outcome = [15, 25, 35, 36, 72, 89, 106, 45];

        $this->assertEquals($outcome, $this->a->abs()->asArray());
    }

    public function test_square()
    {
        $outcome = [225, 625, 1225, 1296, 5184, 7921, 11236, 2025];

        $this->assertEquals($outcome, $this->a->square()->asArray());
    }

    public function test_pow()
    {
        $outcome = [-3375, 15625, 42875, -46656, -373248, 704969, 1191016, 91125];

        $this->assertEquals($outcome, $this->a->pow(3)->asArray());
    }

    public function test_sqrt()
    {
        $outcome = [
            2., 2.5495097567963922, 1.70293863659264, 4.47213595499958,
            1.61245154965971, 3.449637662132068,
        ];

        $this->assertEquals($outcome, $this->c->sqrt()->asArray());
    }

    public function test_exp()
    {
        $outcome = [
            3.0590232050182605E-7, 72004899337.38577, 1586013452313427.8,
            2.319522830243574E-16, 5.380186160021159E-32, 4.489612819174324E+38,
            1.0844638552900169E+46, 3.4934271057485013E+19,
        ];

        $this->assertEquals($outcome, $this->a->exp()->asArray());
    }

    public function test_log()
    {
        $outcome = [
            1.3862943611198906, 1.8718021769015913, 1.0647107369924282,
            2.995732273553991, 0.9555114450274363, 2.4765384001174837,
        ];

        $this->assertEquals($outcome, $this->c->log()->asArray());
    }

    public function test_scalar_subtract()
    {
        $outcome = [-25, 15, 25, -46, -82, 79, 96, 35];

        $this->assertEquals($outcome, $this->a->subtractScalar(10)->asArray());
    }

    public function test_l1_norm()
    {
        $this->assertEquals(423., $this->a->l1Norm());
    }

    public function test_l2_norm()
    {
        $this->assertEquals(172.4441938715247, $this->a->l2Norm());
    }

    public function test_max_norm()
    {
        $this->assertEquals(106., $this->a->maxNorm());
    }
}
