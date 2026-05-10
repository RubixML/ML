<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Datasets\Generators\Agglomerate;

use NumPower;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Generators\Blob\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate\Agglomerate;

#[Group('Generators')]
#[CoversClass(Agglomerate::class)]
class AgglomerateTest extends TestCase
{
    protected const int DATASET_SIZE = 30;

    protected const array WEIGHTS = [1.0, 0.5];

    protected Agglomerate $generator;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'one' => new Blob(
                    center: [-5.0, 3.0],
                    stdDev: 0.2
                ),
                'two' => new Blob(
                    center: [5.0, -3.0],
                    stdDev: 0.2
                ),
            ],
            weights: self::WEIGHTS
        );
    }

    #[Test]
    #[TestDox('Returns normalized weights')]
    public function weights() : void
    {
        $weights = NumPower::divide(NumPower::array(self::WEIGHTS), 1.5)->toArray();

        self::assertEquals(
            ['one' => $weights[0], 'two' => $weights[1]],
            $this->generator->weights()
        );
    }

    #[Test]
    #[TestDox('Returns dimensions')]
    public function dimensions() : void
    {
        self::assertEquals(2, $this->generator->dimensions());
    }

    #[Test]
    #[TestDox('Generates a labeled dataset')]
    public function generate() : void
    {
        $dataset = $this->generator->generate(self::DATASET_SIZE);

        self::assertInstanceOf(Labeled::class, $dataset);
        self::assertInstanceOf(Dataset::class, $dataset);

        self::assertCount(self::DATASET_SIZE, $dataset);
        self::assertEquals(['one', 'two'], $dataset->possibleOutcomes());
    }
}
