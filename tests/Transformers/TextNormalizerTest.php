<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\TextNormalizer;
use PHPUnit\Framework\TestCase;

class TextNormalizerTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\TextNormalizer
     */
    protected $transformer;

    public function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['The quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of Coke'],
            ['with a Dandy   ubrella'],
        ]);

        $this->transformer = new TextNormalizer(true);
    }

    public function test_build_transformer() : void
    {
        $this->assertInstanceOf(TextNormalizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform() : void
    {
        $this->dataset->apply($this->transformer);
    
        $outcome = [
            ['the quick brown fox jumped over the lazy man sitting at a bus'
                . ' stop drinking a can of coke'],
            ['with a dandy ubrella'],
        ];
    
        $this->assertEquals($outcome, $this->dataset->samples());
    }
}
