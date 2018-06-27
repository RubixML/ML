<?php

namespace Rubix\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Spherical;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;

class SphericalTest extends TestCase
{
    protected $kernel;

    public function setUp()
    {
        $this->kernel = new Spherical();
    }

    public function test_build_distance_function()
    {
        $this->assertTrue($this->kernel instanceof Spherical);
        $this->assertTrue($this->kernel instanceof Distance);
    }

    public function test_compute_distance()
    {
        $this->assertEquals(866.71 , round($this->kernel->compute(['x' => 2, 'y' => 3], ['x' => 7, 'y' => 9]), 2));
    }
}
