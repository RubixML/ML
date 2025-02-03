<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers\Seeders;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Clusterers\Seeders\Random;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

#[Group('Seeders')]
#[CoversClass(Random::class)]
class RandomTest extends TestCase
{
    protected Agglomerate $generator;

    protected Random $seeder;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'red' => new Blob(
                    center: [255, 0, 0],
                    stdDev: 30.0
                ),
                'green' => new Blob(
                    center: [0, 128, 0],
                    stdDev: 10.0
                ),
                'blue' => new Blob(
                    center: [0, 0, 255],
                    stdDev: 20.0
                ),
            ],
            weights: [3, 3, 4]
        );

        $this->seeder = new Random();
    }

    public function testSeed() : void
    {
        $dataset = $this->generator->generate(100);

        $seeds = $this->seeder->seed(dataset: $dataset, k: 3);

        $this->assertCount(3, $seeds);
    }
}
