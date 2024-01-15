<?php

namespace Rubix\ML\Tests\Clusterers\Seeders;

use Rubix\ML\Clusterers\Seeders\KMC2;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

/**
 * @group Seeders
 * @covers \Rubix\ML\Clusterers\Seeders\KMC2
 */
class KMC2Test extends TestCase
{
    /**
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var KMC2
     */
    protected $seeder;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 30.0),
            'green' => new Blob([0, 128, 0], 10.0),
            'blue' => new Blob([0, 0, 255], 20.0),
        ], [3, 3, 4]);

        $this->seeder = new KMC2(50, new Euclidean());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(KMC2::class, $this->seeder);
        $this->assertInstanceOf(Seeder::class, $this->seeder);
    }

    /**
     * @test
     */
    public function seed() : void
    {
        $dataset = $this->generator->generate(100);

        $seeds = $this->seeder->seed($dataset, 3);

        $this->assertCount(3, $seeds);
    }
}
