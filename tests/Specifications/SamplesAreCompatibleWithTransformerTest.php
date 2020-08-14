<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\L1Normalizer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Specifications\Specification;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer
 */
class SamplesAreCompatibleWithTransformerTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param \Rubix\ML\Specifications\Specification $spec
     * @param bool $expected
     */
    public function passes(Specification $spec, bool $expected) : void
    {
        $this->assertSame($expected, $spec->passes());
    }

    /**
     * @return \Generator<array>
     */
    public function passesProvider() : Generator
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
}
