package corr.ai.framework.common.exception.enums;

import corr.ai.framework.common.exception.ErrorCode;

/**
 * 全局错误码枚举
 * 0-999 系统异常编码保留
 *
 * 一般情况下，使用 HTTP 响应状态码 https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Status
 * 虽然说，HTTP 响应状态码作为业务使用表达能力偏弱，但是使用在系统层面还是非常不错的
 * 比较特殊的是，因为之前一直使用 0 作为成功，就不使用 200 啦。
 *
 * @author CorrAi
 */
public interface GlobalErrorCodeConstants {

    ErrorCode SUCCESS = new ErrorCode(0, "success");

    // ========== 客户端错误段 ==========

    ErrorCode BAD_REQUEST = new ErrorCode(400, "The request parameters are incorrect.");
    ErrorCode UNAUTHORIZED = new ErrorCode(401, "Account not logged in");
    ErrorCode FORBIDDEN = new ErrorCode(403, "No permission for this operation");
    ErrorCode NOT_FOUND = new ErrorCode(404, "Request not found");
    ErrorCode METHOD_NOT_ALLOWED = new ErrorCode(405, "Incorrect request method");
    ErrorCode LOCKED = new ErrorCode(423, "Request failed, please try again later"); // 并发请求，不允许
    ErrorCode TOO_MANY_REQUESTS = new ErrorCode(429, "The request is too frequent, please try again later");

    // ========== 服务端错误段 ==========

    ErrorCode INTERNAL_SERVER_ERROR = new ErrorCode(500, "System abnormality");
    ErrorCode NOT_IMPLEMENTED = new ErrorCode(501, "Function not implemented/not enabled");
    ErrorCode ERROR_CONFIGURATION = new ErrorCode(502, "Wrong configuration item");

    // ========== 自定义错误段 ==========
    ErrorCode REPEATED_REQUESTS = new ErrorCode(900, "Repeat request, please try again later"); // 重复请求
    ErrorCode DEMO_DENY = new ErrorCode(901, "Demo mode, write operation disabled");

    ErrorCode UNKNOWN = new ErrorCode(999, "Unknown error");

}
