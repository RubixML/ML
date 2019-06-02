<?php

namespace Rubix\ML\Tests\Other\Loggers;

use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Other\Loggers\Logger;
use PHPUnit\Framework\TestCase;
use Psr\Log\LoggerInterface;
use Psr\Log\LogLevel;

class ScreenTest extends TestCase
{
    protected $logger;

    public function setUp()
    {
        $this->logger = new Screen('default', Screen::DEFAULT_TIMESTAMP_FORMAT);
    }

    public function test_build_logger()
    {
        $this->assertInstanceOf(Screen::class, $this->logger);
        $this->assertInstanceOf(Logger::class, $this->logger);
        $this->assertInstanceOf(LoggerInterface::class, $this->logger);
    }

    public function test_log()
    {
        $this->expectOutputRegex('/\b(default.INFO: test)\b/');

        $this->logger->log(LogLevel::INFO, 'test');
    }
}
