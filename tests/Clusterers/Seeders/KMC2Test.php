<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers\Seeders;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Clusterers\Seeders\KMC2;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

#[Group('Seeders')]
#[CoversClass(KMC2::class)]
class KMC2Test extends TestCase
{
    protected Agglomerate $generator;

    protected KMC2 $seeder;

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

        $this->seeder = new KMC2(m: 50, kernel: new Euclidean());
    }

    public function testSeed() : void
    {
        $dataset = $this->generator->generate(100);

        $seeds = $this->seeder->seed(dataset: $dataset, k: 3);

        $this->assertCount(3, $seeds);
    }
}
