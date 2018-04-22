<?php

use Rubix\Engine\Dataset;
use Rubix\Engine\Transformers\TextNormalizer;
use PHPUnit\Framework\TestCase;

class TextNormalizerTest extends TestCase
{
    protected $preprocessor;

    public function setUp()
    {
        $this->preprocessor = new TextNormalizer();
    }

    public function test_build_blanket_character_filter()
    {
        $this->assertInstanceOf(TextNormalizer::class, $this->preprocessor);
    }

    public function test_fit_dataset()
    {
        $this->preprocessor->fit(new Dataset([]));

        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $data = [
            [' The first step  is  to  establish that something is possible.  Then probabiility       will occur.'],
        ];

        $this->preprocessor->transform($data);

        $this->assertEquals([
            ['the first step is to establish that something is possible. then probabiility will occur.'],
        ], $data);
    }
}
