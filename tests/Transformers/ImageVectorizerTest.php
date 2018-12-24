<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\ImageVectorizer;
use PHPUnit\Framework\TestCase;

class ImageVectorizerTest extends TestCase
{
    protected $transformer;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = Unlabeled::quick([
            [imagecreatefromjpeg(__DIR__ . '/../space.jpg')],
        ]);

        $this->transformer = new ImageVectorizer([3, 3], true, 'gd');
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(ImageVectorizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_get_dimensionality()
    {
        $this->assertEquals(27, $this->transformer->dimensions());
    }

    public function test_transform()
    {
        $this->dataset->apply($this->transformer);
    
        $outcome = [
            [22, 35, 60, 53, 66, 102, 29, 44, 73, 36, 49, 79, 45, 57, 89, 21,
            32, 56, 44, 53, 85, 43, 49, 75, 12, 18, 34],
        ];
    
        $this->assertEquals($outcome, $this->dataset->samples());
    }
}
