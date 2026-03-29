<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Datasets\Generators\Hyperplane;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Generators\Hyperplane\Hyperplane;
use Rubix\ML\Datasets\Labeled;

#[Group('Generators')]
#[CoversClass(Hyperplane::class)]
class HyperplaneTest extends TestCase
{
    protected Hyperplane $generator;

    protected function setUp() : void
    {
        $this->generator = new Hyperplane(coefficients: [0.001, -4.0, 12], intercept: 5.0);
    }

    #[Test]
    #[TestDox('Returns the correct number of dimensions')]
    public function dimensions() : void
    {
        self::assertEquals(3, $this->generator->dimensions());
    }

    #[Test]
    #[TestDox('Can generate a labeled dataset')]
    public function generate() : void
    {
        $dataset = $this->generator->generate(30);

        self::assertInstanceOf(Labeled::class, $dataset);
        self::assertInstanceOf(Dataset::class, $dataset);

        self::assertCount(30, $dataset);

        self::assertSame([30, 3], $dataset->shape());

        $samples = $dataset->samples();
        $labels = $dataset->labels();

        self::assertCount(30, $samples);
        self::assertCount(30, $labels);

        foreach ($labels as $label) {
            self::assertIsFloat($label);
            self::assertGreaterThanOrEqual(-1.0, $label);
            self::assertLessThanOrEqual(1.0, $label);
        }

        foreach ($samples as $i => $sample) {
            self::assertCount(3, $sample);

            foreach ($sample as $value) {
                self::assertIsFloat($value);
            }

            $y = $labels[$i];

            $yFromFeature2 = ($sample[1] / -4.0) - 5.0;
            $yFromFeature3 = ($sample[2] / 12.0) - 5.0;

            self::assertEqualsWithDelta($y, $yFromFeature2, 0.2);
            self::assertEqualsWithDelta($y, $yFromFeature3, 0.2);
        }
    }
}
