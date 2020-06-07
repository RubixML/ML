<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\StopWordFilter;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\StopWordFilter
 */
class StopWordFilterTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\StopWordFilter
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['the quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of coke'],
            ['with a dandy umbrella'],
            ['salle à manger'],
        ]);

        $this->transformer = new StopWordFilter(['a', 'quick', 'pig', 'à']);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(StopWordFilter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $expected = [
            ['the  brown fox jumped over the lazy man sitting at  bus stop drinking  can of coke'],
            ['with  dandy umbrella'],
            ['salle  manger'],
        ];

        $this->assertEquals($expected, $this->dataset->samples());
    }
}
