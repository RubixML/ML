<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\TextNormalizer;
use PHPUnit\Framework\TestCase;

class TextNormalizerTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $this->transformer = new TextNormalizer();
    }

    public function test_build_blanket_character_filter()
    {
        $this->assertInstanceOf(TextNormalizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_fit_dataset()
    {
        $this->transformer->fit(new Dataset([]));

        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $data = [
            [' The first step  is  to  establish that something is possible.  Then PROBABILITY       will occur.'],
        ];

        $this->transformer->transform($data);

        $this->assertEquals([
            ['the first step is to establish that something is possible. then probability will occur.'],
        ], $data);
    }
}
