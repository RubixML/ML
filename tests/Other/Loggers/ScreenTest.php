<?php

namespace Rubix\ML\Tests\Other\Loggers;

use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Other\Loggers\Logger;
use PHPUnit\Framework\TestCase;
use Psr\Log\LoggerInterface;
use Psr\Log\LogLevel;

/**
 * @group Loggers
 * @covers \Rubix\ML\Other\Loggers\Screen
 */
class ScreenTest extends TestCase
{
    /**
     * @var \Rubix\ML\Other\Loggers\Screen
     */
    protected $logger;

    protected function setUp() : void
    {
        $this->logger = new Screen('default');
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Screen::class, $this->logger);
        $this->assertInstanceOf(Logger::class, $this->logger);
        $this->assertInstanceOf(LoggerInterface::class, $this->logger);
    }

    /**
     * @test
     */
    public function log() : void
    {
        $this->expectOutputRegex('/\b(default.INFO: test)\b/');

        $this->logger->log(LogLevel::INFO, 'test');
    }
}
