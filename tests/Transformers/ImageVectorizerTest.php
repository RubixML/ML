<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\ImageResizer;
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

        $this->transformer = new ImageVectorizer(3);
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(ImageVectorizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform()
    {
        $this->dataset->apply(new ImageResizer(3, 3));

        $this->dataset->apply($this->transformer);
    
        $outcome = [
            [60, 35, 22, 102, 66, 53, 73, 44, 29, 79, 49, 36, 89, 57, 45, 56,
                32, 21, 85, 53, 44, 75, 49, 43, 34, 18, 12],
        ];
    
        $this->assertEquals($outcome, $this->dataset->samples());
    }
}
