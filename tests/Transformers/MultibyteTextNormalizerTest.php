<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\MultibyteTextNormalizer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(MultibyteTextNormalizer::class)]
class MultibyteTextNormalizerTest extends TestCase
{
    protected Unlabeled $dataset;

    protected MultibyteTextNormalizer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new MultibyteTextNormalizer(false);
    }

    #[Test]
    public function transform() : void
    {
        $dataset = Unlabeled::quick(samples: [
            ['The quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of Coke'],
            ['with a Dandy   umbrella'],
            ['Depuis qu’il avait emménagé à côté de chez elle, il y a de ça cinq ans.'],
            ['Working with emoji 🤓'],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            ['the quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of coke'],
            ['with a dandy   umbrella'],
            ['depuis qu’il avait emménagé à côté de chez elle, il y a de ça cinq ans.'],
            ['working with emoji 🤓'],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }

    #[Test]
    public function transformToUppercase() : void
    {
        $transformer = new MultibyteTextNormalizer(true);

        $dataset = Unlabeled::quick(samples: [
            ['déjà vu à côté', 42],
        ]);

        $dataset->apply($transformer);

        $this->assertEquals([['DÉJÀ VU À CÔTÉ', 42]], $dataset->samples());
    }
}
