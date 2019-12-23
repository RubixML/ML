<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\RegexFilter;
use PHPUnit\Framework\TestCase;

class RegexFilterTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\Transformers\RegexFilter
     */
    protected $transformer;

    public function setUp() : void
    {
        $this->dataset = Unlabeled::quick([
            ['I was not proud of what I had learned, but I never doubted that it was worth knowing'],
            ['Too weird to live, too rare to die https://rubixml.com contact@rubixml.com'],
            ['A man who procrastinates in @his choosing will inevitably have his choice made for him by #circumstance'],
        ]);

        $this->transformer = new RegexFilter(RegexFilter::PATTERNS);
    }

    public function test_build_transformer() : void
    {
        $this->assertInstanceOf(RegexFilter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform() : void
    {
        $this->dataset->apply($this->transformer);
    
        $expected = [
            ['I was not proud of what I had learned, but I never doubted that it was worth knowing'],
            ['Too weird to live, too rare to die  '],
            ['A man who procrastinates in  choosing will inevitably have his choice made for him by '],
        ];
    
        $this->assertEquals($expected, $this->dataset->samples());
    }
}
