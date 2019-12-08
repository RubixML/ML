<?php

namespace Rubix\ML\Tests\Clusterers\Seeders;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

class PlusPlusTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Generators\Generator
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Clusterers\Seeders\PlusPlus
     */
    protected $seeder;

    public function setUp() : void
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ], [3, 3, 4]);

        $this->seeder = new PlusPlus(new Euclidean());
    }

    public function test_build_seeder() : void
    {
        $this->assertInstanceOf(PlusPlus::class, $this->seeder);
        $this->assertInstanceOf(Seeder::class, $this->seeder);
    }

    public function test_seed() : void
    {
        $dataset = $this->generator->generate(100);

        $seeds = $this->seeder->seed($dataset, 3);

        $this->assertCount(3, $seeds);
    }
}
