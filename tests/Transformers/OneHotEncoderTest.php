<?php

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Transformers\OneHotEncoder;
use PHPUnit\Framework\TestCase;

class OneHotEncoderTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $data = new Dataset([
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
        ]);

        $this->transformer = new OneHotEncoder();

        $this->transformer->fit($data);
    }

    public function test_build_one_hot_vectorizer()
    {
        $this->assertInstanceOf(OneHotEncoder::class, $this->transformer);
    }

    public function test_transform_dataset()
    {
        $data = [
            ['nice', 'furry', 'loner'],
            ['warm', 'cuddly', 'pink'],
        ];

        $this->transformer->transform($data);

        $this->assertEquals([
            [1, 1, 0, 0, 1, 0],
            [0, 0, 0 ,0, 0, 0],
        ], $data);
    }

    public function test_encode_sample()
    {
        $vector = $this->transformer->encode(['nice', 'rough', 'loner']);

        $this->assertEquals([1, 0, 0, 0, 1, 1], $vector);
    }
}
