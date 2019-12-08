<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\StopWordFilter;
use PHPUnit\Framework\TestCase;

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

    public function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['the quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of coke'],
            ['with a dandy umbrella'],
        ]);

        $this->transformer = new StopWordFilter(['a', 'quick', 'pig']);
    }

    public function test_build_transformer() : void
    {
        $this->assertInstanceOf(StopWordFilter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform() : void
    {
        $this->dataset->apply($this->transformer);
    
        $outcome = [
            ['the  brown fox jumped over the lazy man sitting at  bus stop drinking  can of coke'],
            ['with  dandy umbrella'],
        ];
    
        $this->assertEquals($outcome, $this->dataset->samples());
    }
}
