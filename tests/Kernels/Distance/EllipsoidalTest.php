<?php

namespace Rubix\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Ellipsoidal;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;

class EllipsoidalTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new Ellipsoidal();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->kernel instanceof Ellipsoidal);
        $this->assertTrue($this->kernel instanceof Distance);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(0.61, round($this->kernel->compute(['x' => 2, 'y' => 3, 'z' => 5], ['x' => 7, 'y' => 9, 'z' => 4]), 2));
    }
}
