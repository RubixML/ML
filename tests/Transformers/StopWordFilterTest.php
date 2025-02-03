<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\StopWordFilter;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(StopWordFilter::class)]
class StopWordFilterTest extends TestCase
{
    protected StopWordFilter $transformer;

    protected function setUp() : void
    {
        $this->transformer = new StopWordFilter(['a', 'quick', 'pig', 'à']);
    }

    public function testTransform() : void
    {
        $dataset = Unlabeled::quick(samples: [
            ['the quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of coke'],
            ['with a dandy umbrella'],
            ['salle à manger'],
        ]);

        $dataset->apply($this->transformer);

        $expected = [
            ['the  brown fox jumped over the lazy man sitting at  bus stop drinking  can of coke'],
            ['with  dandy umbrella'],
            ['salle  manger'],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
