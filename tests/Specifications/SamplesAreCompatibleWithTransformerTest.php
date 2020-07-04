<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\L1Normalizer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer
 */
class SamplesAreCompatibleWithTransformerTest extends TestCase
{
    /**
     * @test
     * @dataProvider checkProvider
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Transformers\Transformer $transformer
     * @param bool $valid
     */
    public function check(Dataset $dataset, Transformer $transformer, bool $valid) : void
    {
        if (!$valid) {
            $this->expectException(InvalidArgumentException::class);
        }

        SamplesAreCompatibleWithTransformer::check($dataset, $transformer);

        $this->assertTrue($valid);
    }

    /**
     * @return \Generator<array>
     */
    public function checkProvider() : Generator
    {
        yield [
            Unlabeled::quick([
                [6.0, -1.1, 5, 'college'],
            ]),
            new L1Normalizer(),
            false,
        ];

        yield [
            Unlabeled::quick([
                [1, 2, 3, 4, 5]
            ]),
            new L1Normalizer(),
            true,
        ];

        yield [
            Unlabeled::quick([
                [6.0, -1.1, 5, 'college'],
            ]),
            new OneHotEncoder(),
            true,
        ];

        yield [
            Unlabeled::quick([
                [6.0, -1.1, 5, 'college'],
            ]),
            new TfIdfTransformer(),
            false,
        ];
    }
}
