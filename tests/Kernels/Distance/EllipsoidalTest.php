<?php

namespace Rubix\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Ellipsoidal;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;

class EllipsoidalTest extends TestCase
{
    protected $kernel;

    protected $a;

    protected $b;

    protected $c;

    public function setUp()
    {
        $this->a = ['x' => 2, 'y' => 1, 'z' => 4];
        $this->b = ['x' => 7, 'y' => 9, 'z' => 4];
        $this->c = ['x' => 2, 'y' => 2, 'z' => 3];

        $this->kernel = new Ellipsoidal();
    }

    public function test_build_distance_kernel()
    {
        $this->assertInstanceOf(Ellipsoidal::class, $this->kernel);
        $this->assertInstanceOf(Distance::class, $this->kernel);
    }

    public function test_compute_distance()
    {
        $distance1 = $this->kernel->compute($this->a, $this->b);
        $distance2 = $this->kernel->compute($this->a, $this->c);
        $distance3 = $this->kernel->compute($this->b, $this->c);

        $this->assertEquals(0.7893120976935766, $distance1);
        $this->assertEquals(0.3089247840430205, $distance2);
        $this->assertEquals(0.4881858109030444, $distance3);
    }
}
