<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\BlanketCharacterFilter;
use PHPUnit\Framework\TestCase;

class BlanketCharacterFilterTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $this->transformer = new BlanketCharacterFilter([',']);

        $this->transformer->fit(new Dataset(['some text']));
    }

    public function test_build_blanket_character_filter()
    {
        $this->assertInstanceOf(BlanketCharacterFilter::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_fit_dataset()
    {
        $this->assertTrue(true);
    }

    public function test_transform_dataset()
    {
        $data = [
            ['I like cooking, my family, and pets.'],
        ];

        $this->transformer->transform($data);

        $this->assertEquals([
            ['I like cooking my family and pets.']
        ], $data);
    }
}
