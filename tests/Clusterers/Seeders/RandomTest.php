<?php

namespace Rubix\ML\Tests\Clusterers\Seeders;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Clusterers\Seeders\Seeder;
use Rubix\ML\Clusterers\Seeders\Random;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

class RandomTest extends TestCase
{
    protected $generator;

    protected $seeder;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ], [3, 3, 4]);

        $this->seeder = new Random();
    }

    public function test_build_seeder()
    {
        $this->assertInstanceOf(Random::class, $this->seeder);
        $this->assertInstanceOf(Seeder::class, $this->seeder);
    }

    public function test_seed()
    {
        $dataset = $this->generator->generate(100);

        $seeds = $this->seeder->seed($dataset, 3);

        $this->assertCount(3, $seeds);
    }
}
