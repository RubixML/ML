<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\FloatTypeConverter;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(FloatTypeConverter::class)]
class FloatTypeConverterTest extends TestCase
{
    protected FloatTypeConverter $transformer;

    protected function setUp() : void
    {
        $this->transformer = new FloatTypeConverter();
    }

    #[Test]
    public function transform() : void
    {
        $dataset = new Unlabeled(samples: [
            [1, 2, 3.0, '4', 'NAN', 'a'],
            [4, 5, 6.0, '5.0', 'INF', 'b'],
            [7, 8, 9.0, '6', '-INF', 'c'],
        ]);

        $this->assertEquals(
            [
                DataType::categorical(),
                DataType::categorical(),
                DataType::continuous(),
                DataType::categorical(),
                DataType::categorical(),
                DataType::categorical(),
            ],
            $dataset->featureTypes()
        );

        $dataset->apply($this->transformer);

        $samples = $dataset->samples();

        $this->assertEquals([1.0, 2.0, 3.0, 4.0, 'a'], [
            $samples[0][0], $samples[0][1], $samples[0][2], $samples[0][3], $samples[0][5],
        ]);

        $this->assertIsFloat($samples[0][0]);
        $this->assertIsFloat($samples[0][1]);
        $this->assertIsFloat($samples[0][2]);
        $this->assertIsFloat($samples[0][3]);
        $this->assertNan($samples[0][4]);
        $this->assertIsString($samples[0][5]);

        $this->assertEquals([4.0, 5.0, 6.0, 5.0, 'b'], [
            $samples[1][0], $samples[1][1], $samples[1][2], $samples[1][3], $samples[1][5],
        ]);

        $this->assertIsFloat($samples[1][0]);
        $this->assertIsFloat($samples[1][3]);
        $this->assertInfinite($samples[1][4]);
        $this->assertTrue($samples[1][4] > 0);

        $this->assertEquals([7.0, 8.0, 9.0, 6.0, 'c'], [
            $samples[2][0], $samples[2][1], $samples[2][2], $samples[2][3], $samples[2][5],
        ]);

        $this->assertInfinite($samples[2][4]);
        $this->assertTrue($samples[2][4] < 0);

        $this->assertEquals(
            [
                DataType::continuous(),
                DataType::continuous(),
                DataType::continuous(),
                DataType::continuous(),
                DataType::continuous(),
                DataType::categorical(),
            ],
            $dataset->featureTypes()
        );

        $this->assertEquals('Float Type Converter', (string) $this->transformer);
    }
}
