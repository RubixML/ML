<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\L1Normalizer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(SamplesAreCompatibleWithTransformer::class)]
class SamplesAreCompatibleWithTransformerTest extends TestCase
{
    /**
     * @return Generator<mixed[]>
     */
    public static function passesProvider() : Generator
    {
        yield [
            SamplesAreCompatibleWithTransformer::with(
                Unlabeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ]),
                new L1Normalizer()
            ),
            false,
        ];

        yield [
            SamplesAreCompatibleWithTransformer::with(
                Unlabeled::quick([
                    [1, 2, 3, 4, 5]
                ]),
                new L1Normalizer()
            ),
            true,
        ];

        yield [
            SamplesAreCompatibleWithTransformer::with(
                Unlabeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ]),
                new OneHotEncoder()
            ),
            true,
        ];

        yield [
            SamplesAreCompatibleWithTransformer::with(
                Unlabeled::quick([
                    [6.0, -1.1, 5, 'college'],
                ]),
                new TfIdfTransformer()
            ),
            false,
        ];
    }

    /**
     * @param SamplesAreCompatibleWithTransformer $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(SamplesAreCompatibleWithTransformer $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}
