<?php

use Rubix\Engine\Datasets\Unlabeled;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Transformers\OneHotEncoder;
use PHPUnit\Framework\TestCase;

class OneHotEncoderTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $data = new Unlabeled([
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
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform_dataset()
    {
        $data = [
            ['nice', 'furry', 'loner'],
            ['warm', 'cuddly', 'pink'],
        ];

        $this->transformer->transform($data);

        $this->assertEquals([
            [1, 0, 1, 0, 0, 1],
            [0, 0, 0 ,0, 0, 0],
        ], $data);
    }
}
