<?php

namespace Rubix\Tests\Kernels\Distance;

use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Kernels\Distance\Distance;
use PHPUnit\Framework\TestCase;

class DiagonalTest extends TestCase
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

        $this->kernel = new Diagonal();
    }

    public function test_build_distance_kernel()
    {
        $this->assertInstanceOf(Diagonal::class, $this->kernel);
        $this->assertInstanceOf(Distance::class, $this->kernel);
    }

    public function test_compute_distance()
    {
        $distance1 = $this->kernel->compute($this->a, $this->b);
        $distance2 = $this->kernel->compute($this->a, $this->c);
        $distance3 = $this->kernel->compute($this->b, $this->c);

        $this->assertEquals(8.0, $distance1);
        $this->assertEquals(1.0, $distance2);
        $this->assertEquals(7.0, $distance3);
    }
}
