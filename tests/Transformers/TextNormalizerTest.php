<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\TextNormalizer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(TextNormalizer::class)]
class TextNormalizerTest extends TestCase
{
    protected TextNormalizer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new TextNormalizer(true);
    }

    public function testTransform() : void
    {
        $dataset = Unlabeled::quick(samples: [
            ['The quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of Coke'],
            ['with a Dandy   umbrella'],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            ['THE QUICK BROWN FOX JUMPED OVER THE LAZY MAN SITTING AT A BUS STOP DRINKING A CAN OF COKE'],
            ['WITH A DANDY   UMBRELLA'],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
