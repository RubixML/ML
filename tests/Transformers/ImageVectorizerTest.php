<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\ImageResizer;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[RequiresPhpExtension('gd')]
#[CoversClass(ImageVectorizer::class)]
class ImageVectorizerTest extends TestCase
{
    protected ImageVectorizer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new ImageVectorizer(false);
    }

    #[Test]
    public function fitTransform() : void
    {
        $dataset = Unlabeled::quick([
            [imagecreatefrompng('tests/test.png'), 'something else'],
        ]);

        $dataset->apply(new ImageResizer(3, 3));

        $dataset->apply($this->transformer);

        $expected = [
            ['something else', 46, 51, 66, 130, 135, 134, 118, 119, 116, 25, 26, 45, 149, 154, 154, 180,
                183, 170, 39, 39, 54, 77, 80, 89, 141, 140, 132],
        ];

        $this->assertEqualsWithDelta($expected, $dataset->samples(), 1.0);
    }

    #[Test]
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = [
            [imagecreate(1, 1), 'something else'],
        ];

        $this->transformer->transform($samples);
    }
}
