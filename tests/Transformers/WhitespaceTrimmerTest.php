<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\WhitespaceTrimmer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\WhitespaceTrimmer
 */
class WhitespaceTrimmerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\WhitespaceTrimmer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['The quick brown fox jumped  over  the lazy man sitting at a bus'
                . ' stop drinking a can of     Coke'],
            [' with a Dandy   umbrella '],
        ]);

        $this->transformer = new WhitespaceTrimmer();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(WhitespaceTrimmer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    /**
     * @test
     */
    public function transform() : void
    {
        $this->dataset->apply($this->transformer);

        $outcome = [
            ['The quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of Coke'],
            ['with a Dandy umbrella'],
        ];

        $this->assertEquals($outcome, $this->dataset->samples());
    }
}
