<?php

use Rubix\Engine\Datasets\Unlabeled;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\TextNormalizer;
use PHPUnit\Framework\TestCase;

class TextNormalizerTest extends TestCase
{
    protected $transformer;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            [' The first step  is  to  establish that something is possible.'
                . '  Then PROBABILITY       will occur.'],
        ]);

        $this->transformer = new TextNormalizer();
    }

    public function test_build_text_normalizer()
    {
        $this->assertInstanceOf(TextNormalizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_dataset()
    {
        $this->dataset->transform($this->transformer);

        $this->assertEquals([
            ['the first step is to establish that something is possible. then probability will occur.'],
        ], $this->dataset->samples());
    }
}
